import os
import glob
import json
import pickle
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

# CSV 컬럼 구조:
# [I_meas, HF_energy, HF_rms, hf_norm, arc_flag, arc_detect, T_meas, mode]
REQUIRED_COLS = [
    "I_meas",
    "HF_energy",
    "HF_rms",
    "hf_norm",
    "arc_flag",
    "arc_detect",
    "T_meas",
    "mode",
]

# 시퀀스 입력:
# [I_meas, HF_energy, HF_rms, hf_norm, arc_flag, arc_detect, T_meas]
SEQ_DIM = 7

# 라벨 컬럼:
# mode
LABEL_COL = 7

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
    x_seq: [T, 7]
    columns:
      0: I_meas
      1: HF_energy
      2: HF_rms
      3: hf_norm
      4: arc_flag
      5: arc_detect
      6: T_meas

    return: [18]
    """
    I = x_seq[:, 0]
    HF_energy = x_seq[:, 1]
    HF_rms = x_seq[:, 2]
    hf_norm = x_seq[:, 3]
    arc_flag = x_seq[:, 4]
    arc_detect = x_seq[:, 5]
    T = x_seq[:, 6]

    dI = np.diff(I, prepend=I[0])
    dT = np.diff(T, prepend=T[0])
    dHF = np.diff(HF_energy, prepend=HF_energy[0])

    feats = [
        # current features
        I.mean(),
        I.std(),
        np.sqrt(np.mean(I ** 2)),

        # HF energy features
        HF_energy.mean(),
        HF_energy.std(),
        HF_energy.max(),

        # HF rms features
        HF_rms.mean(),
        HF_rms.std(),

        # normalized HF feature
        hf_norm.mean(),
        hf_norm.std(),
        hf_norm.max(),

        # rule-based detector intermediate/final features
        arc_flag.mean(),
        arc_flag.max(),

        arc_detect.mean(),
        arc_detect.max(),

        # temperature features
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
      xxx_mode0_xxx.csv
      xxx_mode1_xxx.csv
      xxx_mode2_xxx.csv
      xxx_mode3_xxx.csv
      xxx_mode4_xxx.csv
      xxx_mode7_xxx.csv
    """
    name = os.path.basename(filename).lower()

    # mode10 같은 이름이 있을 경우를 대비해 긴 숫자부터 체크
    for mode in [7, 6, 5, 4, 3, 2, 1, 0]:
        if f"mode{mode}" in name:
            return mode

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

        train_part, temp_part = train_test_split(
            mode_files,
            test_size=(1.0 - TRAIN_RATIO),
            random_state=RANDOM_STATE,
            shuffle=True
        )

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


def validate_columns(df: pd.DataFrame, filename: str):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"필수 컬럼이 없습니다: {filename}\n"
            f"missing={missing}\n"
            f"current_columns={list(df.columns)}"
        )


def load_selected_csvs(file_list):
    arrays = []
    file_names = []

    for f in file_list:
        df = pd.read_csv(f)
        validate_columns(df, f)

        # 필요한 컬럼만 명시적으로 선택
        df = df[REQUIRED_COLS]

        # mode 검증
        file_mode = infer_mode_from_filename(f)
        csv_modes = df["mode"].dropna().unique()

        if len(csv_modes) == 0:
            raise ValueError(f"mode 컬럼이 비어 있습니다: {f}")

        # 기본적으로 한 파일은 한 mode라고 가정
        if len(csv_modes) == 1:
            csv_mode = int(csv_modes[0])
            if csv_mode != file_mode:
                raise ValueError(
                    f"파일명 mode와 CSV 내부 mode가 다릅니다: {f}\n"
                    f"filename_mode={file_mode}, csv_mode={csv_mode}"
                )

        # boolean 또는 문자열이 섞였을 가능성을 방지하기 위해 숫자 변환
        for col in REQUIRED_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df.isna().any().any():
            nan_cols = df.columns[df.isna().any()].tolist()
            raise ValueError(
                f"CSV에 NaN 또는 숫자로 변환 불가능한 값이 있습니다: {f}\n"
                f"problem_columns={nan_cols}"
            )

        arr = df.to_numpy(dtype=np.float32)

        if arr.shape[1] != len(REQUIRED_COLS):
            raise ValueError(
                f"CSV 컬럼 수가 예상과 다릅니다: {f}, "
                f"shape={arr.shape}, expected_cols={len(REQUIRED_COLS)}"
            )

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
        X = arr[:, :SEQ_DIM].astype(np.float32)
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

    print(f"[INFO] Train X_seq shape : {X_seq_train.shape}")
    print(f"[INFO] Train X_feat shape: {X_feat_train.shape}")
    print(f"[INFO] Val   X_seq shape : {X_seq_val.shape}")
    print(f"[INFO] Val   X_feat shape: {X_feat_val.shape}")
    print(f"[INFO] Test  X_seq shape : {X_seq_test.shape}")
    print(f"[INFO] Test  X_feat shape: {X_feat_test.shape}")

    print(f"[INFO] Train class dist: {class_distribution(y_train)}")
    print(f"[INFO] Val   class dist: {class_distribution(y_val)}")
    print(f"[INFO] Test  class dist: {class_distribution(y_test)}")

    print(f"[DEBUG] y_train unique: {np.unique(y_train)}")
    print(f"[DEBUG] y_val unique  : {np.unique(y_val)}")
    print(f"[DEBUG] y_test unique : {np.unique(y_test)}")

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
        "required_cols": REQUIRED_COLS,
        "seq_dim": SEQ_DIM,
        "seq_columns": REQUIRED_COLS[:SEQ_DIM],
        "label_col": REQUIRED_COLS[LABEL_COL],
        "feat_dim": int(X_feat_train.shape[1]),
        "feature_names": [
            "I_mean",
            "I_std",
            "I_rms",
            "HF_energy_mean",
            "HF_energy_std",
            "HF_energy_max",
            "HF_rms_mean",
            "HF_rms_std",
            "hf_norm_mean",
            "hf_norm_std",
            "hf_norm_max",
            "arc_flag_mean",
            "arc_flag_max",
            "arc_detect_mean",
            "arc_detect_max",
            "T_mean",
            "dT_mean",
            "dT_max",
        ],
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
