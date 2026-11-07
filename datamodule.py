from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import lightning as L

from helpers import DEVICES, load_day, normalize


class VRMotionDataset(Dataset):
    def __init__(self, x, y, device_ids=None):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        if device_ids is None:
            self.device_ids = None
        else:
            self.device_ids = torch.from_numpy(device_ids)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.device_ids is None:
            return self.x[idx], self.y[idx]
        return self.x[idx], self.y[idx], self.device_ids[idx]


class VRBiometricDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir="Dataset/MotionsNumpy",
        modality="full",
        devices=None,
        batch_size=32,
        num_workers=0,
        val_ratio=0.2,
        seed=42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.modality = modality
        self.devices = devices if devices is not None else DEVICES
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.n_features = None
        self.n_classes = None
        self.seq_len = None

    def setup(self, stage=None):
        train_x, train_y, train_d = load_day(
            self.data_dir, 1, self.devices, self.modality
        )
        test_x, test_y, test_d = load_day(
            self.data_dir, 2, self.devices, self.modality
        )
        train_x, test_x = normalize(train_x, test_x)

        self.n_features = train_x.shape[-1]
        self.seq_len = train_x.shape[1]
        self.n_classes = int(train_y.max()) + 1

        full_train = VRMotionDataset(train_x, train_y, train_d)
        n_val = int(len(full_train) * self.val_ratio)
        n_train = len(full_train) - n_val
        gen = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.val_ds = random_split(full_train, [n_train, n_val], generator=gen)
        self.test_ds = VRMotionDataset(test_x, test_y, test_d)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
