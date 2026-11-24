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


EULER_SEQ = "xyz"


def encode_rotation(rot):
    """Katy Eulera (3 sensory x 3 katy) -> reprezentacja 6D (3 sensory x 6).

    Surowe katy maja dwa problemy:
    - zawijaja sie na granicy +-pi (np. 3.1 -> -3.1 w jednym kroku),
    - gimbal lock: gdy rot_y zbliza sie do +-pi/2, rot_x i rot_z przeskakuja
      razem, mimo ze sama orientacja zmienia sie plynnie.
    Dlatego katy zamieniamy na macierz rotacji i bierzemy jej dwie pierwsze
    kolumny (reprezentacja 6D, Zhou i in. 2019). Jest ona ciagla.
    Kolejnosc "xyz" (zewnetrzna) wybrana empirycznie: przy niej orientacja
    helmu i lewego kontrolera zmienia sie miedzy klatkami o < 0.2 rad.
    """
    from scipy.spatial.transform import Rotation

    shape = rot.shape[:-1]
    n_sensors = rot.shape[-1] // 3
    eulers = rot.reshape(-1, 3).astype(np.float64)
    matrices = Rotation.from_euler(EULER_SEQ, eulers).as_matrix()
    six_d = matrices[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    return six_d.reshape(*shape, n_sensors * 6)


def flatten_session(arr, modality="full"):
    modality_indices(modality)
    n_subj, n_throws, n_t, _ = arr.shape
    parts = []
    if modality in ("position", "full"):
        parts.append(arr[..., POS_IDX])
    if modality in ("rotation", "full"):
        parts.append(encode_rotation(arr[..., ROT_IDX]))
    x = np.concatenate(parts, axis=-1)
    x = x.reshape(n_subj * n_throws, n_t, x.shape[-1])
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
