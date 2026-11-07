import numpy as np

DEVICES = ["Quest", "Vive", "Cosmos"]

POS_IDX = [0, 1, 2, 7, 8, 9, 14, 15, 16]
ROT_IDX = [3, 4, 5, 10, 11, 12, 17, 18, 19]
FULL_IDX = POS_IDX + ROT_IDX

MODALITY_IDX = {
    "position": POS_IDX,
    "rotation": ROT_IDX,
    "full": FULL_IDX,
}


def modality_indices(modality):
    if modality not in MODALITY_IDX:
        raise ValueError(f"Nieznana modalnosc: {modality}")
    return MODALITY_IDX[modality]


def load_session(data_dir, device, day):
    path = data_dir / f"{device}{day}.npy"
    return np.load(path)


def flatten_session(arr, modality="full"):
    idx = modality_indices(modality)
    n_subj, n_throws, n_t, _ = arr.shape
    x = arr[:, :, :, idx].reshape(n_subj * n_throws, n_t, len(idx))
    y = np.repeat(np.arange(n_subj), n_throws)
    return x.astype(np.float32), y.astype(np.int64)


def load_day(data_dir, day, devices=None, modality="full"):
    if devices is None:
        devices = DEVICES
    xs, ys, ds = [], [], []
    for device in devices:
        arr = load_session(data_dir, device, day)
        x, y = flatten_session(arr, modality)
        xs.append(x)
        ys.append(y)
        ds.append(np.full(len(y), DEVICES.index(device), dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ds)


def normalize(train_x, test_x):
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std
