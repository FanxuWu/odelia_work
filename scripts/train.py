import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.data_module import MRIDataModule
from src.model_3d import MRI3DClassifier


def main():
    csv_path = "classification_index.csv"
    batch_size = 1
    num_workers = 0
    max_epochs = 3

    data_module = MRIDataModule(
        csv_path=csv_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = MRI3DClassifier()

    logger = TensorBoardLogger(
        save_dir="logs",
        name="mri_3d",
        default_hp_metric=False,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=False,
        # deterministic is disabled to avoid RuntimeError with avg_pool3d_backward
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
