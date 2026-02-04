import os
import argparse
import numpy as np
from scipy.io import loadmat

def zscore_fit(X_list):
    concat = np.concatenate(X_list, axis=0)
    mu = concat.mean(axis=0, keepdims=True)
    sd = concat.std(axis=0, keepdims=True) + 1e-8
    return mu, sd

def zscore_apply(X, mu, sd):
    return (X - mu) / sd

def sliding_window(X, y_mode, run_id, win_len, stride):
    T, C = X.shape
    xs, ys, rids = [], [], []
    for s in range(0, T - win_len + 1, stride):
        xs.append(X[s:s+win_len])
        ys.append(y_mode)
        rids.append(run_id)
    if len(xs) == 0:
        return (np.empty((0, win_len, C), np.float32),
                np.empty((0,), np.int64),
                np.empty((0,), np.int64))
    return (np.stack(xs).astype(np.float32),
            np.array(ys, dtype=np.int64),
            np.array(rids, dtype=np.int64))

def stratified_run_split(runs, seed=42, n_train_per_class=None, n_val_per_class=1, n_test_per_class=1):
    """
    runs: list of run objects, each has .mode and .X
    각 클래스(mode)마다 run을 섞은 뒤
    test/val 각각 n개씩 떼고 나머지 train으로.
    """
    rng = np.random.default_rng(seed)
    idx_by_c = {c: [] for c in [0,1,2,3]}

    for i, r in enumerate(runs):
        idx_by_c[int(r.mode)].append(i)

    for c in idx_by_c:
        rng.shuffle(idx_by_c[c])
        if len(idx_by_c[c]) < (n_val_per_class + n_test_per_class + 1):
            raise ValueError(f"class {c} run 수가 너무 적습니다: {len(idx_by_c[c])}")

    tr, va, te = [], [], []
    for c in [0,1,2,3]:
        te_c = idx_by_c[c][:n_test_per_class]
        va_c = idx_by_c[c][n_test_per_class:n_test_per_class+n_val_per_class]
        rest = idx_by_c[c][n_test_per_class+n_val_per_class:]

        if n_train_per_class is not None:
            if len(rest) < n_train_per_class:
                raise ValueError(f"class {c} train run 부족: need {n_train_per_class}, have {len(rest)}")
            tr_c = rest[:n_train_per_class]
        else:
            tr_c = rest

        te += te_c
        va += va_c
        tr += tr_c

    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return np.array(tr), np.array(va), np.array(te)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_mat", type=str, required=True)
    ap.add_argument("--out_npz", type=str, required=True)
    ap.add_argument("--win_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--normalize", type=str, default="zscore", choices=["none","zscore"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_val_per_class", type=int, default=1)
    ap.add_argument("--n_test_per_class", type=int, default=1)
    args = ap.parse_args()

    mat = loadmat(args.in_mat, squeeze_me=True, struct_as_record=False)
    flat = mat["flat"]
    runs = flat.runs
    if not isinstance(runs, (list, np.ndarray)):
        runs = [runs]
    else:
        runs = list(runs)

    tr_idx, va_idx, te_idx = stratified_run_split(
        runs, seed=args.seed,
        n_val_per_class=args.n_val_per_class,
        n_test_per_class=args.n_test_per_class
    )

    def get_runs(indices):
        Xs, ys, rids = [], [], []
        for rid, i in enumerate(indices):
            r = runs[int(i)]
            X = np.array(r.X, dtype=np.float32)
            y = int(r.mode)
            Xs.append(X)
            ys.append(y)
            rids.append(int(i))  # run index as run_id
        return Xs, ys, rids

    Xtr_runs, ytr_runs, tr_runids = get_runs(tr_idx)
    Xva_runs, yva_runs, va_runids = get_runs(va_idx)
    Xte_runs, yte_runs, te_runids = get_runs(te_idx)

    # normalize (fit on train runs)
    if args.normalize == "zscore":
        mu, sd = zscore_fit(Xtr_runs)
        Xtr_runs = [zscore_apply(x, mu, sd) for x in Xtr_runs]
        Xva_runs = [zscore_apply(x, mu, sd) for x in Xva_runs]
        Xte_runs = [zscore_apply(x, mu, sd) for x in Xte_runs]
    else:
        mu = None; sd = None

    def make_windows(X_runs, y_runs, run_ids):
        Xw_all, yw_all, rid_all = [], [], []
        for X, y, rid in zip(X_runs, y_runs, run_ids):
            Xw, yw, rids = sliding_window(X, y, rid, args.win_len, args.stride)
            if len(Xw) > 0:
                Xw_all.append(Xw); yw_all.append(yw); rid_all.append(rids)
        return (np.concatenate(Xw_all, axis=0),
                np.concatenate(yw_all, axis=0),
                np.concatenate(rid_all, axis=0))

    Xtr, ytr, rid_tr = make_windows(Xtr_runs, ytr_runs, tr_runids)
    Xva, yva, rid_va = make_windows(Xva_runs, yva_runs, va_runids)
    Xte, yte, rid_te = make_windows(Xte_runs, yte_runs, te_runids)

    print("[INFO] shapes")
    print("  train:", Xtr.shape, ytr.shape, "unique runs:", len(np.unique(rid_tr)))
    print("  val  :", Xva.shape, yva.shape, "unique runs:", len(np.unique(rid_va)))
    print("  test :", Xte.shape, yte.shape, "unique runs:", len(np.unique(rid_te)))
    print("  test class counts:", {c:int((yte==c).sum()) for c in [0,1,2,3]})

    np.savez(
        args.out_npz,
        X_train=Xtr, y_train=ytr, runid_train=rid_tr,
        X_val=Xva, y_val=yva, runid_val=rid_va,
        X_test=Xte, y_test=yte, runid_test=rid_te,
        win_len=args.win_len, stride=args.stride,
        normalize=args.normalize,
        mu=(mu.astype(np.float32) if mu is not None else None),
        sd=(sd.astype(np.float32) if sd is not None else None),
        split_seed=args.seed,
        n_val_per_class=args.n_val_per_class,
        n_test_per_class=args.n_test_per_class
    )
    print(f"[SAVED] {args.out_npz}")

if __name__ == "__main__":
    main()
