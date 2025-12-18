import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.networks.nets import DenseNet121

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
)


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

        # --- Task2: buffers for validation predictions/targets ---
        self.val_preds = []
        self.val_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"].long()
        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        # on_step + on_epoch: makes TensorBoard curves clearer
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"].long()
        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        # log val loss (epoch-level)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # --- Task2: collect predictions/targets for confusion matrix & report ---
        preds = torch.argmax(logits, dim=1)
        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(labels.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        # If nothing collected (edge case), skip
        if len(self.val_preds) == 0 or len(self.val_targets) == 0:
            return

        preds = torch.cat(self.val_preds).numpy()
        targets = torch.cat(self.val_targets).numpy()

        # Clear buffers for next epoch
        self.val_preds.clear()
        self.val_targets.clear()

        num_classes = int(self.hparams.num_classes)
        labels_list = list(range(num_classes))

        # --- Confusion Matrix ---
        cm = confusion_matrix(targets, preds, labels=labels_list)

        # --- Metrics ---
        acc = accuracy_score(targets, preds)
        prec_macro = precision_score(targets, preds, average="macro", zero_division=0)
        rec_macro = recall_score(targets, preds, average="macro", zero_division=0)

        self.log("val_acc", acc, prog_bar=True, logger=True)
        self.log("val_precision_macro", prec_macro, prog_bar=False, logger=True)
        self.log("val_recall_macro", rec_macro, prog_bar=False, logger=True)

        # --- Write to TensorBoard (figure + text) ---
        if self.logger is not None and hasattr(self.logger, "experiment"):
            tb = self.logger.experiment

            # 1) Confusion matrix figure
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(cm)
            plt.title("Confusion Matrix (Validation)")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.colorbar()

            tick_names = [str(i) for i in labels_list]
            plt.xticks(labels_list, tick_names)
            plt.yticks(labels_list, tick_names)

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, int(cm[i, j]), ha="center", va="center")

            plt.tight_layout()
            tb.add_figure("confusion_matrix/val", fig, global_step=self.current_epoch)
            plt.close(fig)

            # 2) Classification report text
            report = classification_report(
                targets, preds,
                labels=labels_list,
                digits=4,
                zero_division=0
            )
            tb.add_text(
                "classification_report/val",
                f"<pre>{report}</pre>",
                global_step=self.current_epoch
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
