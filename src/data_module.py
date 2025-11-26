
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
)


class OdeliaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_path: str = "classification_index.csv",
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = None
        self.val_set = None
        self.transform = None

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)

        items = []
        for _, row in df.iterrows():
            items.append(
                {
                    "image": row["volume_path"],
                    "label": int(row["label"]),
                }
            )

        self.transform = Compose(
            [
                LoadImaged(keys="image", image_only=True),
                EnsureChannelFirstd(keys="image"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

        dataset = Dataset(data=items, transform=self.transform)

        # 为了简单，先用同一个 dataset 做 train / val
        self.train_set = dataset
        self.val_set = dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def _smoke_test():
    dm = OdeliaDataModule(
        csv_path="classification_index.csv",
        batch_size=1,
        num_workers=0,
    )
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    img = batch["image"]
    label = batch["label"]

    print("image shape:", img.shape)
    print("label:", label)
    print("dtype:", img.dtype)


if __name__ == "__main__":
    _smoke_test()
