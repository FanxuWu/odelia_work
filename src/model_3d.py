import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.networks.nets import DenseNet121


class MRI3DClassifier(pl.LightningModule):
    def __init__(self, lr=1e-4, num_classes=3):
        super().__init__()
        self.save_hyperparameters()

        # MONAI 3D DenseNet
        self.model = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes
        )

        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"].long()
        preds = self(images)
        loss = F.cross_entropy(preds, labels)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"].long()
        preds = self(images)
        loss = F.cross_entropy(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
