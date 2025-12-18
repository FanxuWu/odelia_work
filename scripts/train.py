import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data_module import MRIDataModule
from src.model_3d import MRI3DClassifier


def main():
    # ---- basic configs ----
    csv_path = "classification_index.csv"
    batch_size = 1
    num_workers = 4

    # Task 1 requirement: train for at least 10 epochs
    max_epochs = 10

    # ---- data module ----
    data_module = MRIDataModule(
        csv_path=csv_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ---- model ----
    model = MRI3DClassifier()

    # ---- TensorBoard logger ----
    logger = TensorBoardLogger(
        save_dir="logs",
        name="mri_3d",
        default_hp_metric=False,
    )

    # ---- checkpoint callback
    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints/mri_3d",
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # ---- trainer ----
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[ckpt_cb],
        enable_checkpointing=True,
        enable_progress_bar=True,
        # deterministic is disabled to avoid RuntimeError with avg_pool3d_backward
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
