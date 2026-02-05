# make_dataset.py
import os
import argparse
import numpy as np
from scipy.io import loadmat

def zscore_fit(X_list):
    # X_list: list of (T,C) arrays from train only
    concat = np.concatenate(X_list, axis=0)  # (sumT, C)
    mu = concat.mean(axis=0, keepdims=True)
    sd = concat.std(axis=0, keepdims=True) + 1e-8
    return mu, sd

def zscore_apply(X, mu, sd):
    return (X - mu) / sd

def sliding_window(X, y_mode, win_len, stride):
    # X: (T,C)
    T, C = X.shape
    xs, ys = [], []
    for s in range(0, T - win_len + 1, stride):
        xs.append(X[s:s+win_len])
        ys.append(y_mode)
    if len(xs) == 0:
        return np.empty((0, win_len, C), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)

def time_split_by_run(runs, ratios=(0.7, 0.15, 0.15), seed=42):
    # run 단위로 split (데이터 누수 방지에 매우 유리)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(runs))
    rng.shuffle(idx)

    n = len(runs)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])

    tr = idx[:n_tr]
    va = idx[n_tr:n_tr+n_va]
    te = idx[n_tr+n_va:]
    return tr, va, te

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_mat", type=str, default="dataset_out/dataset_flat.mat")
    ap.add_argument("--out_npz", type=str, default="train_dataset.npz")
    ap.add_argument("--win_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--normalize", type=str, default="zscore", choices=["none","zscore"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    args = ap.parse_args()

    mat = loadmat(args.in_mat, squeeze_me=True, struct_as_record=False)
    flat = mat["flat"]

    runs = flat.runs
    # runs가 단일이면 iterable 처리
    if not isinstance(runs, (list, np.ndarray)):
        runs = [runs]
    else:
        runs = list(runs)

    # run 단위 split
    tr_idx, va_idx, te_idx = time_split_by_run(runs, ratios=tuple(args.split), seed=args.seed)

    def collect_run_arrays(indices):
        Xs, ys, metas = [], [], []
        for i in indices:
            r = runs[i]
            X = np.array(r.X, dtype=np.float32)  # (T,C)
            y = int(r.mode)
            Xs.append(X)
            ys.append(y)
            metas.append({"mode": y, "label": str(r.label)})
        return Xs, ys, metas

    X_train_runs, y_train_runs, _ = collect_run_arrays(tr_idx)
    X_val_runs,   y_val_runs,   _ = collect_run_arrays(va_idx)
    X_test_runs,  y_test_runs,  _ = collect_run_arrays(te_idx)

    # 정규화 (train 기준)
    if args.normalize == "zscore":
        mu, sd = zscore_fit(X_train_runs)
        X_train_runs = [zscore_apply(x, mu, sd) for x in X_train_runs]
        X_val_runs   = [zscore_apply(x, mu, sd) for x in X_val_runs]
        X_test_runs  = [zscore_apply(x, mu, sd) for x in X_test_runs]
    else:
        mu = None
        sd = None

    # sliding window로 (N,win,C) 생성
    def make_windows(X_runs, y_runs):
        Xw_all, yw_all = [], []
        for X, y in zip(X_runs, y_runs):
            Xw, yw = sliding_window(X, y, args.win_len, args.stride)
            if len(Xw) > 0:
                Xw_all.append(Xw)
                yw_all.append(yw)
        if len(Xw_all) == 0:
            return np.empty((0, args.win_len, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)
        return np.concatenate(Xw_all, axis=0), np.concatenate(yw_all, axis=0)

    Xtr, ytr = make_windows(X_train_runs, y_train_runs)
    Xva, yva = make_windows(X_val_runs,   y_val_runs)
    Xte, yte = make_windows(X_test_runs,  y_test_runs)

    print("[INFO] shapes")
    print("  train:", Xtr.shape, ytr.shape)
    print("  val  :", Xva.shape, yva.shape)
    print("  test :", Xte.shape, yte.shape)

    # 저장
    np.savez(
        args.out_npz,
        X_train=Xtr, y_train=ytr,
        X_val=Xva,   y_val=yva,
        X_test=Xte,  y_test=yte,
        win_len=args.win_len, stride=args.stride,
        normalize=args.normalize,
        mu=(mu.astype(np.float32) if mu is not None else None),
        sd=(sd.astype(np.float32) if sd is not None else None),
    )
    print(f"[SAVED] {args.out_npz}")

if __name__ == "__main__":
    main()
