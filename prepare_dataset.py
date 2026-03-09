import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = "./dataset_out"
SAVE_DIR = "./prepared_dataset"
WINDOW = 200
STRIDE = 50
RANDOM_STATE = 42

os.makedirs(SAVE_DIR, exist_ok=True)


def compute_engineered_features(x_seq: np.ndarray) -> np.ndarray:
    # x_seq: [T, 3] -> columns: I, HF, T
    I = x_seq[:, 0]
    HF = x_seq[:, 1]
    T = x_seq[:, 2]

    dT = np.diff(T, prepend=T[0])

    feats = [
        I.mean(),
        I.std(),
        np.sqrt(np.mean(I**2)),
        HF.mean(),
        HF.std(),
        HF.max(),
        T.mean(),
        dT.mean(),
        dT.max(),
    ]
    return np.array(feats, dtype=np.float32)


def load_csvs():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError("CSV 파일이 없습니다.")
    arrays = []
    for f in files:
        df = pd.read_csv(f)
        arrays.append(df.to_numpy(dtype=np.float32))
    return arrays


def make_sequences(arrays):
    X_seq_all, X_feat_all, y_all = [], [], []

    for arr in arrays:
        X = arr[:, :3]   # I, HF, T
        y = arr[:, 3].astype(np.int64)

        for start in range(0, len(X) - WINDOW + 1, STRIDE):
            end = start + WINDOW
            x_seq = X[start:end]
            label = int(y[end - 1])

            X_seq_all.append(x_seq)
            X_feat_all.append(compute_engineered_features(x_seq))
            y_all.append(label)

    return (
        np.array(X_seq_all, dtype=np.float32),
        np.array(X_feat_all, dtype=np.float32),
        np.array(y_all, dtype=np.int64),
    )


def normalize_data(X_seq, X_feat):
    # sequence normalize
    n, t, f = X_seq.shape
    seq_scaler = StandardScaler()
    X_seq_2d = X_seq.reshape(-1, f)
    X_seq_norm = seq_scaler.fit_transform(X_seq_2d).reshape(n, t, f)

    feat_scaler = StandardScaler()
    X_feat_norm = feat_scaler.fit_transform(X_feat)

    return X_seq_norm, X_feat_norm, seq_scaler, feat_scaler


def main():
    arrays = load_csvs()
    X_seq, X_feat, y = make_sequences(arrays)

    X_seq_train, X_seq_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
        X_seq, X_feat, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    X_seq_train, X_seq_val, X_feat_train, X_feat_val, y_train, y_val = train_test_split(
        X_seq_train, X_feat_train, y_train,
        test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )

    X_seq_train, X_feat_train, seq_scaler, feat_scaler = normalize_data(X_seq_train, X_feat_train)

    # val/test scaler 적용
    X_seq_val = seq_scaler.transform(X_seq_val.reshape(-1, 3)).reshape(X_seq_val.shape)
    X_seq_test = seq_scaler.transform(X_seq_test.reshape(-1, 3)).reshape(X_seq_test.shape)
    X_feat_val = feat_scaler.transform(X_feat_val)
    X_feat_test = feat_scaler.transform(X_feat_test)

    np.save(os.path.join(SAVE_DIR, "X_seq_train.npy"), X_seq_train)
    np.save(os.path.join(SAVE_DIR, "X_seq_val.npy"), X_seq_val)
    np.save(os.path.join(SAVE_DIR, "X_seq_test.npy"), X_seq_test)

    np.save(os.path.join(SAVE_DIR, "X_feat_train.npy"), X_feat_train)
    np.save(os.path.join(SAVE_DIR, "X_feat_val.npy"), X_feat_val)
    np.save(os.path.join(SAVE_DIR, "X_feat_test.npy"), X_feat_test)

    np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(SAVE_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)

    print("저장 완료")
    print("X_seq_train:", X_seq_train.shape)
    print("X_feat_train:", X_feat_train.shape)
    print("y_train:", y_train.shape)


if __name__ == "__main__":
    main()