import argparse
import csv
import os
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn


# ----------------------
# Models (GRU/LSTM + Temporal Attention)
# ----------------------
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):  # (B,T,H)
        score = self.v(torch.tanh(self.proj(h)))          # (B,T,1)
        alpha = torch.softmax(score, dim=1).squeeze(-1)   # (B,T)
        ctx = torch.sum(h * alpha.unsqueeze(-1), dim=1)   # (B,H)
        return ctx, alpha


class RNNAttnClassifier(nn.Module):
    def __init__(self, rnn_type="gru", input_dim=3, hidden_dim=128, num_classes=4):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        self.attn = TemporalAttention(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        h, _ = self.rnn(x)            # (B,T,H)
        ctx, alpha = self.attn(h)     # (B,H), (B,T)
        logits = self.head(ctx)       # (B,K)
        return logits, alpha


def load_ckpt(model: nn.Module, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer_prob(model: nn.Module, window_tc: np.ndarray, device: str):
    """
    window_tc: (T,C) float32
    returns: (K,) numpy probabilities
    """
    x = torch.tensor(window_tc, dtype=torch.float32).unsqueeze(0).to(device)  # (1,T,C)
    logits, _ = model(x)
    prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return prob


# ----------------------
# Streaming buffer
# ----------------------
class RingBuffer:
    def __init__(self, win_len: int, channels: int):
        self.win_len = win_len
        self.channels = channels
        self.buf = np.zeros((win_len, channels), dtype=np.float32)
        self.filled = 0

    def push(self, sample: np.ndarray):
        self.buf[:-1] = self.buf[1:]
        self.buf[-1] = sample
        self.filled = min(self.win_len, self.filled + 1)

    def ready(self):
        return self.filled >= self.win_len

    def window(self):
        return self.buf.copy()


# ----------------------
# Multi-class 2-stage decision logic + Alarm Levels + Hysteresis
# ----------------------
class MultiClassTwoStageDecision:
    """
    (C) 멀티클래스 게이팅 최적화:
      - Stage1에서 의심(gate1=True)된 클래스만 Stage2 검증/투표에 반영
      - gate1=False 클래스는 hist에 0을 넣어 알람이 유지되지 않게 함

    (B) Alarm Level (3-level):
      - CAUTION: vote_sum >= 1
      - WARNING: vote_sum >= M  (알람 ON과 동치)
      - DANGER : vote_sum >= danger_M  또는 (p2 >= danger_p2 and vote_sum >= M)
    """
    def __init__(
        self,
        class_names,
        alarm_classes=("loose", "arc", "overcurrent"),
        theta1=0.70,
        theta2=0.90,
        N=8,
        M=3,
        clear_N=8,
        clear_M=1,
        danger_extra=2,      # danger_M = min(N, M + danger_extra)
        danger_p2=0.98,      # p2가 매우 높으면 danger로 올림(단, vote>=M 조건)
    ):
        self.class_names = list(class_names)
        self.K = len(self.class_names)

        self.alarm_idxs = [self.class_names.index(c) for c in alarm_classes]
        self.theta1 = self._to_per_class(theta1, default=0.70)
        self.theta2 = self._to_per_class(theta2, default=0.90)

        self.N = int(N)
        self.M = int(M)
        self.clear_N = int(clear_N)
        self.clear_M = int(clear_M)

        self.danger_M = int(min(self.N, self.M + int(danger_extra)))
        self.danger_p2 = float(danger_p2)

        self.hist = {c: deque(maxlen=self.N) for c in self.alarm_idxs}
        self.alarm_state = {c: False for c in self.alarm_idxs}

    def _to_per_class(self, x, default):
        arr = np.full((self.K,), float(default), dtype=np.float32)
        if isinstance(x, (int, float)):
            arr[:] = float(x)
            return arr
        if isinstance(x, str):
            vals = [float(v.strip()) for v in x.split(",")]
            if len(vals) == 1:
                arr[:] = vals[0]
            elif len(vals) == self.K:
                arr[:] = vals
            else:
                raise ValueError(f"Threshold list must have length 1 or {self.K}")
            return arr
        vals = list(x)
        if len(vals) == 1:
            arr[:] = float(vals[0])
        elif len(vals) == self.K:
            arr[:] = np.array(vals, dtype=np.float32)
        else:
            raise ValueError(f"Threshold list must have length 1 or {self.K}")
        return arr

    def compute_gates(self, p1: np.ndarray):
        """Stage1 게이팅 결과를 클래스별로 반환"""
        gated = {}
        for c in self.alarm_idxs:
            gated[c] = (float(p1[c]) >= float(self.theta1[c]))
        return gated

    def update(self, p1: np.ndarray, p2: np.ndarray | None):
        events = []
        status = []

        gated = self.compute_gates(p1)

        # update hist per alarm class
        for c in self.alarm_idxs:
            p1c = float(p1[c])
            p2c = None if p2 is None else float(p2[c])

            if gated[c] and (p2 is not None):
                pass2 = (p2c >= float(self.theta2[c]))
                self.hist[c].append(1 if pass2 else 0)
            else:
                pass2 = False
                self.hist[c].append(0)

            vote_sum = int(sum(self.hist[c]))
            vote_len = int(len(self.hist[c]))

            # alarm ON/OFF
            if (not self.alarm_state[c]) and (vote_sum >= self.M):
                self.alarm_state[c] = True
                events.append((self.class_names[c], "ALARM_ON"))
            elif self.alarm_state[c]:
                recent = list(self.hist[c])[-self.clear_N:] if vote_len >= self.clear_N else list(self.hist[c])
                if sum(recent) <= self.clear_M:
                    self.alarm_state[c] = False
                    events.append((self.class_names[c], "ALARM_OFF"))

            # (B) alarm level
            if vote_sum >= self.danger_M:
                level = "DANGER"
            elif (p2c is not None) and (p2c >= self.danger_p2) and (vote_sum >= self.M):
                level = "DANGER"
            elif vote_sum >= self.M:
                level = "WARNING"
            elif vote_sum >= 1:
                level = "CAUTION"
            else:
                level = "NORMAL"

            status.append({
                "class": self.class_names[c],
                "p1": p1c,
                "p2": p2c,
                "gate1": bool(gated[c]),
                "pass2": bool(pass2),
                "vote_sum": vote_sum,
                "vote": f"{vote_sum}/{self.N}",
                "alarm": bool(self.alarm_state[c]),
                "level": level,
            })

        return status, events


# ----------------------
# CSV Logger (A)
# ----------------------
class CsvLogger:
    def __init__(self, path: str, class_names, alarm_classes):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)

        # header
        header = [
            "t_wall", "t_rel", "step",
            "lat_stage1_ms", "lat_stage2_ms", "lat_total_ms",
            "stage2_called", "events"
        ]
        # per class columns (alarm classes only, normal excluded by default)
        for c in alarm_classes:
            header += [
                f"p1_{c}", f"p2_{c}", f"gate1_{c}", f"pass2_{c}",
                f"vote_sum_{c}", f"alarm_{c}", f"level_{c}"
            ]
        self.w.writerow(header)
        self.start_wall = time.time()
        self.start_perf = time.perf_counter()
        self.alarm_classes = list(alarm_classes)

    def log(self, step, lat1, lat2, latt, stage2_called, events_str, status_list):
        now_wall = time.time()
        now_rel = time.perf_counter() - self.start_perf

        # build lookup
        st_map = {st["class"]: st for st in status_list}

        row = [
            now_wall, now_rel, step,
            lat1, (lat2 if lat2 is not None else ""), latt,
            int(stage2_called), events_str
        ]
        for c in self.alarm_classes:
            st = st_map[c]
            row += [
                f"{st['p1']:.6f}",
                ("" if st["p2"] is None else f"{st['p2']:.6f}"),
                int(st["gate1"]),
                int(st["pass2"]),
                int(st["vote_sum"]),
                int(st["alarm"]),
                st["level"],
            ]
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except:
            pass


# ----------------------
# Main
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt_stage1", required=True, help="LSTM+Attn (fast detector)")
    ap.add_argument("--ckpt_stage2", required=True, help="GRU+Attn (stable verifier)")
    ap.add_argument("--win_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)

    # thresholds can be scalar or "a,b,c,d"
    ap.add_argument("--theta1", default="0.70", help="scalar or 4 values for [normal,loose,arc,over]")
    ap.add_argument("--theta2", default="0.90", help="scalar or 4 values for [normal,loose,arc,over]")

    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--clear_N", type=int, default=8)
    ap.add_argument("--clear_M", type=int, default=1)

    # danger level parameters
    ap.add_argument("--danger_extra", type=int, default=2, help="danger_M = min(N, M + danger_extra)")
    ap.add_argument("--danger_p2", type=float, default=0.98, help="if p2>=danger_p2 and vote>=M -> DANGER")

    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--print_every", type=int, default=10)

    # (A) csv logging
    ap.add_argument("--log_csv", type=str, default="", help="e.g., .\\logs\\stream_2stage_log.csv (optional)")

    args = ap.parse_args()

    class_names = ["normal", "loose", "arc", "overcurrent"]
    alarm_classes = ("loose", "arc", "overcurrent")  # normal은 알람 제외(원하면 추가)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    data = np.load(args.npz)
    X_test = data["X_test"].astype(np.float32)
    if X_test.shape[1] != args.win_len:
        raise ValueError(f"X_test window length {X_test.shape[1]} != --win_len {args.win_len}")

    # Stream simulation source (실제 센서에서는 ADC sample로 rb.push 하면 됩니다)
    stream_seq = X_test[0]  # (T,C), normalized
    rb = RingBuffer(args.win_len, 3)

    stage1 = RNNAttnClassifier(rnn_type="lstm", input_dim=3, hidden_dim=128, num_classes=4)
    stage2 = RNNAttnClassifier(rnn_type="gru",  input_dim=3, hidden_dim=128, num_classes=4)
    stage1 = load_ckpt(stage1, args.ckpt_stage1, device)
    stage2 = load_ckpt(stage2, args.ckpt_stage2, device)

    dec = MultiClassTwoStageDecision(
        class_names=class_names,
        alarm_classes=alarm_classes,
        theta1=args.theta1,
        theta2=args.theta2,
        N=args.N,
        M=args.M,
        clear_N=args.clear_N,
        clear_M=args.clear_M,
        danger_extra=args.danger_extra,
        danger_p2=args.danger_p2
    )

    logger = None
    if args.log_csv.strip():
        logger = CsvLogger(args.log_csv.strip(), class_names, alarm_classes)
        print(f"[INFO] CSV logging to: {args.log_csv.strip()}")

    # warmup
    for _ in range(args.warmup):
        for i in range(args.win_len):
            rb.push(stream_seq[i])
        _ = infer_prob(stage1, rb.window(), device)
        _ = infer_prob(stage2, rb.window(), device)

    # run steps (seconds is approximate; stride~0.25s 가정으로 steps=4*seconds)
    steps = max(1, int(args.seconds * 4))
    stride = args.stride

    t_s1, t_s2, t_total = [], [], []
    stage2_called_cnt = 0
    alarm_events_total = 0

    step_idx = 0
    for step in range(1, steps + 1):
        # push stride samples
        for i in range(stride):
            rb.push(stream_seq[(step_idx + i) % args.win_len])
        step_idx += stride

        if not rb.ready():
            continue

        w = rb.window()

        t0 = time.perf_counter()

        # stage1
        ts1_0 = time.perf_counter()
        p1 = infer_prob(stage1, w, device)
        ts1_1 = time.perf_counter()
        lat1 = (ts1_1 - ts1_0) * 1000.0
        t_s1.append(lat1)

        # (C) multiclass gating: call stage2 only if any alarm class is gated by stage1
        gated = dec.compute_gates(p1)
        gate_any = any(gated[c] for c in dec.alarm_idxs)

        p2 = None
        lat2 = None
        stage2_called = False
        if gate_any:
            stage2_called = True
            stage2_called_cnt += 1
            ts2_0 = time.perf_counter()
            p2 = infer_prob(stage2, w, device)
            ts2_1 = time.perf_counter()
            lat2 = (ts2_1 - ts2_0) * 1000.0
            t_s2.append(lat2)

        status, events = dec.update(p1, p2)
        alarm_events_total += len(events)

        t1 = time.perf_counter()
        latt = (t1 - t0) * 1000.0
        t_total.append(latt)

        events_str = ",".join([f"{c}:{e}" for c, e in events]) if events else ""

        # print
        if (step % args.print_every) == 0:
            pieces = []
            for st in status:
                c = st["class"]
                p2s = "None" if st["p2"] is None else f"{st['p2']:.2f}"
                pieces.append(
                    f"{c}:p1={st['p1']:.2f},p2={p2s},gate={st['gate1']},vote={st['vote']},"
                    f"alarm={st['alarm']},lvl={st['level']}"
                )
            msg = f"[STEP {step:04d}] " + " | ".join(pieces)
            if events:
                msg += "  <-- EVENTS: " + events_str
            print(msg)

        # (A) csv log
        if logger is not None:
            logger.log(
                step=step,
                lat1=lat1,
                lat2=lat2,
                latt=latt,
                stage2_called=stage2_called,
                events_str=events_str,
                status_list=status
            )

    # close logger
    if logger is not None:
        logger.close()

    # stats
    t_s1 = np.array(t_s1, dtype=np.float64)
    t_s2 = np.array(t_s2, dtype=np.float64) if len(t_s2) else np.array([], dtype=np.float64)
    t_total = np.array(t_total, dtype=np.float64)

    def stat(x):
        if len(x) == 0:
            return {"mean": None, "p95": None, "max": None}
        return {"mean": float(x.mean()), "p95": float(np.percentile(x, 95)), "max": float(x.max())}

    s1 = stat(t_s1)
    s2 = stat(t_s2)
    st = stat(t_total)

    print("\n========== Multi-class 2-Stage Summary ==========")
    print(f"[Stage1 LSTM] mean={s1['mean']:.3f} ms | p95={s1['p95']:.3f} ms | max={s1['max']:.3f} ms")
    if s2["mean"] is None:
        print("[Stage2 GRU ] (never called)")
    else:
        print(f"[Stage2 GRU ] mean={s2['mean']:.3f} ms | p95={s2['p95']:.3f} ms | max={s2['max']:.3f} ms")
    print(f"[TOTAL]      mean={st['mean']:.3f} ms | p95={st['p95']:.3f} ms | max={st['max']:.3f} ms")
    print(f"[Stage2 called ratio] {stage2_called_cnt}/{len(t_total)} = {(stage2_called_cnt/max(1,len(t_total)))*100:.1f}%")
    print(f"[Alarm events total] {alarm_events_total}")
    if args.log_csv.strip():
        print(f"[CSV] saved: {args.log_csv.strip()}")
    print("===============================================\n")


if __name__ == "__main__":
    main()
