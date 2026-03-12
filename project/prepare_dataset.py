import os
import glob
import json
import pickle
import re
import numpy as np
import pandas as pd

from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================
# 사용자 설정
# =========================
DATA_DIR = "./dataset_out"
SAVE_ROOT = "./prepared_dataset_multi"

WINDOWS = [64, 128, 256]
STRIDE_MAP = {
    64: 16,
    128: 32,
    256: 64,
}

RANDOM_STATE = 42

# 라벨 방식: "last", "majority", "center"
LABEL_MODE = "last"

# None 이면 purity 체크 안 함
PURITY_THRESHOLD = None

# CSV 컬럼 가정: [I, HF, T, label]
SEQ_DIM = 3
LABEL_COL = 3

# CSV split 비율
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

os.makedirs(SAVE_ROOT, exist_ok=True)


# =========================================================
# Feature Engineering
# =========================================================
def compute_engineered_features(x_seq: np.ndarray) -> np.ndarray:
    """
    x_seq: [T, 3] -> columns: I, HF, T
    return: [9]
    """
    I = x_seq[:, 0]
    HF = x_seq[:, 1]
    T = x_seq[:, 2]

    dT = np.diff(T, prepend=T[0])

    feats = [
        I.mean(),
        I.std(),
        np.sqrt(np.mean(I ** 2)),
        HF.mean(),
        HF.std(),
        HF.max(),
        T.mean(),
        dT.mean(),
        dT.max(),
    ]
    return np.array(feats, dtype=np.float32)


# =========================================================
# CSV Loading / Split
# =========================================================
def infer_mode_from_filename(filename: str):
    """
    파일명에서 mode 추정.
    예:
      normal_01.csv
      loose_03.csv
      arc_07.csv
      overcurrent_10.csv

    필요 시 이 함수만 사용자 파일명에 맞게 수정하시면 됩니다.
    """
    name = os.path.basename(filename).lower()

    if "mode0" in name:
        return 0
    if "mode1" in name:
        return 1
    if "mode2" in name:
        return 2
    if "mode3" in name:
        return 3

    raise ValueError(f"파일명에서 mode를 추정할 수 없습니다: {filename}")


def load_csv_paths():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError(f"CSV 파일이 없습니다: {DATA_DIR}")
    return files


def split_csv_files_by_mode(files):
    """
    CSV 파일 단위로 split.
    각 mode별로 train/val/test 비율 유지.
    """
    mode_to_files = defaultdict(list)
    for f in files:
        mode = infer_mode_from_filename(f)
        mode_to_files[mode].append(f)

    train_files, val_files, test_files = [], [], []

    for mode in sorted(mode_to_files.keys()):
        mode_files = sorted(mode_to_files[mode])

        if len(mode_files) < 3:
            raise ValueError(
                f"mode={mode} 파일 수가 너무 적습니다. "
                f"최소 3개 이상 필요합니다. 현재: {len(mode_files)}"
            )

        # 1차: train vs temp(val+test)
        train_part, temp_part = train_test_split(
            mode_files,
            test_size=(1.0 - TRAIN_RATIO),
            random_state=RANDOM_STATE,
            shuffle=True
        )

        # 2차: val vs test
        # TRAIN=0.8, VAL=0.1, TEST=0.1 이면 temp=0.2 이고 그 반반
        val_part, test_part = train_test_split(
            temp_part,
            test_size=0.5,
            random_state=RANDOM_STATE,
            shuffle=True
        )

        train_files.extend(train_part)
        val_files.extend(val_part)
        test_files.extend(test_part)

    return sorted(train_files), sorted(val_files), sorted(test_files)


def load_selected_csvs(file_list):
    arrays = []
    file_names = []

    for f in file_list:
        df = pd.read_csv(f)
        arr = df.to_numpy(dtype=np.float32)

        if arr.shape[1] < 4:
            raise ValueError(f"CSV 컬럼 수가 부족합니다: {f}, shape={arr.shape}")

        arrays.append(arr)
        file_names.append(os.path.basename(f))

    return arrays, file_names


# =========================================================
# Label policy
# =========================================================
def get_window_label(y_window: np.ndarray, mode: str):
    """
    y_window: [WINDOW]
    mode: "last", "majority", "center"
    """
    y_window = y_window.astype(np.int64)

    if mode == "last":
        label = int(y_window[-1])
    elif mode == "center":
        label = int(y_window[len(y_window) // 2])
    elif mode == "majority":
        counter = Counter(y_window.tolist())
        max_count = max(counter.values())
        candidate_labels = [k for k, v in counter.items() if v == max_count]
        last_label = int(y_window[-1])
        label = last_label if last_label in candidate_labels else int(candidate_labels[0])
    else:
        raise ValueError(f"지원하지 않는 LABEL_MODE: {mode}")

    return label


def check_purity(y_window: np.ndarray, label: int, threshold: float):
    if threshold is None:
        return True
    ratio = np.mean(y_window == label)
    return ratio >= threshold


# =========================================================
# Sequence generation
# =========================================================
def make_sequences(arrays, file_names, window, stride, label_mode="last", purity_threshold=None):
    X_seq_all, X_feat_all, y_all = [], [], []
    meta_all = []

    for arr, fname in zip(arrays, file_names):
        X = arr[:, :SEQ_DIM]
        y = arr[:, LABEL_COL].astype(np.int64)

        if len(X) < window:
            continue

        for start in range(0, len(X) - window + 1, stride):
            end = start + window

            x_seq = X[start:end]
            y_window = y[start:end]

            label = get_window_label(y_window, label_mode)

            if not check_purity(y_window, label, purity_threshold):
                continue

            X_seq_all.append(x_seq)
            X_feat_all.append(compute_engineered_features(x_seq))
            y_all.append(label)

            meta_all.append({
                "file": fname,
                "start": int(start),
                "end": int(end),
                "label": int(label),
            })

    if len(X_seq_all) == 0:
        raise ValueError(
            f"생성된 시퀀스가 없습니다. "
            f"window={window}, stride={stride}, label_mode={label_mode}, purity={purity_threshold}"
        )

    return (
        np.array(X_seq_all, dtype=np.float32),
        np.array(X_feat_all, dtype=np.float32),
        np.array(y_all, dtype=np.int64),
        meta_all,
    )


# =========================================================
# Normalize
# =========================================================
def normalize_data(X_seq_train, X_feat_train):
    """
    train 기준으로 fit
    """
    n, t, f = X_seq_train.shape

    seq_scaler = StandardScaler()
    X_seq_train_2d = X_seq_train.reshape(-1, f)
    X_seq_train_norm = seq_scaler.fit_transform(X_seq_train_2d).reshape(n, t, f)

    feat_scaler = StandardScaler()
    X_feat_train_norm = feat_scaler.fit_transform(X_feat_train)

    return X_seq_train_norm, X_feat_train_norm, seq_scaler, feat_scaler


def apply_seq_scaler(X_seq, seq_scaler):
    n, t, f = X_seq.shape
    return seq_scaler.transform(X_seq.reshape(-1, f)).reshape(n, t, f)


# =========================================================
# Save / Report
# =========================================================
def save_scaler(path, scaler):
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def class_distribution(y):
    counter = Counter(y.tolist())
    total = len(y)
    dist = {}
    for k in sorted(counter.keys()):
        dist[int(k)] = {
            "count": int(counter[k]),
            "ratio": float(counter[k] / total)
        }
    return dist


def save_metadata_json(path, metadata):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def print_file_split_info(train_files, val_files, test_files):
    print("=" * 80)
    print("[INFO] CSV split 결과")
    print(f"  train files: {len(train_files)}")
    print(f"  val files  : {len(val_files)}")
    print(f"  test files : {len(test_files)}")

    print("\n[INFO] train files")
    for f in train_files:
        print(" ", os.path.basename(f))

    print("\n[INFO] val files")
    for f in val_files:
        print(" ", os.path.basename(f))

    print("\n[INFO] test files")
    for f in test_files:
        print(" ", os.path.basename(f))


# =========================================================
# Main process for one window
# =========================================================
def process_one_window(train_arrays, train_names,
                       val_arrays, val_names,
                       test_arrays, test_names,
                       window, stride):

    print("=" * 80)
    print(f"[INFO] WINDOW={window}, STRIDE={stride}, LABEL_MODE={LABEL_MODE}, PURITY={PURITY_THRESHOLD}")

    save_dir = os.path.join(SAVE_ROOT, f"win_{window}")
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # split별로 따로 sequence 생성
    # -----------------------------
    X_seq_train, X_feat_train, y_train, meta_train = make_sequences(
        train_arrays, train_names, window, stride,
        label_mode=LABEL_MODE,
        purity_threshold=PURITY_THRESHOLD,
    )

    X_seq_val, X_feat_val, y_val, meta_val = make_sequences(
        val_arrays, val_names, window, stride,
        label_mode=LABEL_MODE,
        purity_threshold=PURITY_THRESHOLD,
    )

    X_seq_test, X_feat_test, y_test, meta_test = make_sequences(
        test_arrays, test_names, window, stride,
        label_mode=LABEL_MODE,
        purity_threshold=PURITY_THRESHOLD,
    )

    print(f"[INFO] Train X_seq shape: {X_seq_train.shape}")
    print(f"[INFO] Val   X_seq shape: {X_seq_val.shape}")
    print(f"[INFO] Test  X_seq shape: {X_seq_test.shape}")

    print(f"[INFO] Train class dist: {class_distribution(y_train)}")
    print(f"[INFO] Val   class dist: {class_distribution(y_val)}")
    print(f"[INFO] Test  class dist: {class_distribution(y_test)}")

    # -----------------------------
    # normalize (train 기준)
    # -----------------------------
    X_seq_train, X_feat_train, seq_scaler, feat_scaler = normalize_data(X_seq_train, X_feat_train)

    X_seq_val = apply_seq_scaler(X_seq_val, seq_scaler)
    X_seq_test = apply_seq_scaler(X_seq_test, seq_scaler)

    X_feat_val = feat_scaler.transform(X_feat_val)
    X_feat_test = feat_scaler.transform(X_feat_test)

    # -----------------------------
    # 저장
    # -----------------------------
    np.save(os.path.join(save_dir, "X_seq_train.npy"), X_seq_train)
    np.save(os.path.join(save_dir, "X_seq_val.npy"), X_seq_val)
    np.save(os.path.join(save_dir, "X_seq_test.npy"), X_seq_test)

    np.save(os.path.join(save_dir, "X_feat_train.npy"), X_feat_train)
    np.save(os.path.join(save_dir, "X_feat_val.npy"), X_feat_val)
    np.save(os.path.join(save_dir, "X_feat_test.npy"), X_feat_test)

    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "y_val.npy"), y_val)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)

    save_scaler(os.path.join(save_dir, "seq_scaler.pkl"), seq_scaler)
    save_scaler(os.path.join(save_dir, "feat_scaler.pkl"), feat_scaler)

    save_metadata_json(os.path.join(save_dir, "meta_train.json"), meta_train)
    save_metadata_json(os.path.join(save_dir, "meta_val.json"), meta_val)
    save_metadata_json(os.path.join(save_dir, "meta_test.json"), meta_test)

    config = {
        "window": window,
        "stride": stride,
        "label_mode": LABEL_MODE,
        "purity_threshold": PURITY_THRESHOLD,
        "random_state": RANDOM_STATE,
        "seq_dim": SEQ_DIM,
        "feat_dim": 9,
        "num_classes": int(len(np.unique(y_train))),
        "train_distribution": class_distribution(y_train),
        "val_distribution": class_distribution(y_val),
        "test_distribution": class_distribution(y_test),
        "train_files": train_names,
        "val_files": val_names,
        "test_files": test_names,
    }
    save_metadata_json(os.path.join(save_dir, "config.json"), config)

    print(f"[SAVE] {save_dir}")
    print(f"  X_seq_train : {X_seq_train.shape}")
    print(f"  X_seq_val   : {X_seq_val.shape}")
    print(f"  X_seq_test  : {X_seq_test.shape}")
    print(f"  X_feat_train: {X_feat_train.shape}")
    print(f"  y_train     : {y_train.shape}")
    print(f"  train dist  : {class_distribution(y_train)}")
    print(f"  val dist    : {class_distribution(y_val)}")
    print(f"  test dist   : {class_distribution(y_test)}")


# =========================================================
# Main
# =========================================================
def main():
    files = load_csv_paths()

    train_files, val_files, test_files = split_csv_files_by_mode(files)
    print_file_split_info(train_files, val_files, test_files)

    train_arrays, train_names = load_selected_csvs(train_files)
    val_arrays, val_names = load_selected_csvs(val_files)
    test_arrays, test_names = load_selected_csvs(test_files)

    for window in WINDOWS:
        if window not in STRIDE_MAP:
            raise ValueError(f"STRIDE_MAP에 WINDOW={window} 설정이 없습니다.")
        stride = STRIDE_MAP[window]

        process_one_window(
            train_arrays, train_names,
            val_arrays, val_names,
            test_arrays, test_names,
            window, stride
        )

    print("=" * 80)
    print("모든 WINDOW 처리 완료")


if __name__ == "__main__":
    main()