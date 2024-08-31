"""Train a PointNet++ tree segmentor directly and only on the synthetic forest data."""

import os
import pathlib
import tempfile
from typing import Optional

import dotenv
import lightning as L
import torch
import torch_geometric
import torchinfo
from torch import nn

from src.datasets import SyntheticForest
from src.models.pointnet import PointNet2TreeSegmentor


class LitPointNet2TreeSegmentor(L.LightningModule):
    """A PointNet++ tree segmentor lightning module."""

    def __init__(self):
        """Create a new LitPointNet2TreeSegmentor instance."""
        super().__init__()

        self.pointnet = PointNet2TreeSegmentor(num_features=4)

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        """Process a single batch of the training dataset and return the loss."""
        pred = self.pointnet(batch)
        loss = nn.functional.mse_loss(pred.squeeze(), batch.y.float())

        self.logger.log_metrics({"loss/train": loss.item()}, step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        """Process a single batch of the validation dataset and return the loss."""
        pred = self.pointnet(batch)
        loss = nn.functional.mse_loss(pred.squeeze(), batch.y.float())

        self.logger.log_metrics({"loss/val": loss.item()}, step=self.global_step)

    def configure_optimizers(self):
        """Set up and return the optimizers."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class SyntheticForestDataModule(L.LightningDataModule):
    """A data module for the synthetic forest dataset."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        las_features: Optional[list[str]] = None,
    ):
        """Create a new SyntheticForestDataModule instance."""
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.las_features = las_features
        self.transform = torch_geometric.transforms.Compose(
            [
                torch_geometric.transforms.NormalizeScale(),
                torch_geometric.transforms.NormalizeFeatures(),
            ]
        )

    def prepare_data(self):
        """Prepare the data for setup (download, tokenize, etc.) on one device."""
        SyntheticForest(
            root=self.data_dir,
            random_seed=42,
            train_samples=100,
            val_samples=20,
            test_samples=20,
            trees_per_sample=50,
            las_features=self.las_features,
        )

    def setup(self, stage: str):
        """Prepare the data for training (split, transform, etc.) on all devices."""
        if stage == "fit":
            self.train = SyntheticForest(
                root=self.data_dir,
                split="train",
                random_seed=42,
                train_samples=100,
                val_samples=20,
                test_samples=20,
                trees_per_sample=50,
                las_features=self.las_features,
            )
            self.val = SyntheticForest(
                root=self.data_dir,
                split="val",
                random_seed=42,
                train_samples=100,
                val_samples=20,
                test_samples=20,
                trees_per_sample=50,
                las_features=self.las_features,
            )

        if stage == "test":
            raise NotImplementedError()

        if stage == "predict":
            raise NotImplementedError()

    def train_dataloader(self):
        """Set up and return the train data loader."""
        return torch_geometric.loader.DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        """Set up and return the validation data loader."""
        return torch_geometric.loader.DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        """Set up and return the test data loader."""
        raise NotImplementedError()

    def predict_dataloader(self):
        """Set up and return the prediction data loader."""
        raise NotImplementedError()


if __name__ == "__main__":
    if pathlib.Path(".env").exists():
        dotenv.load_dotenv(override=True)

    model = LitPointNet2TreeSegmentor()
    mnist = SyntheticForestDataModule(
        data_dir="data/raw/synthetic_forest",
        batch_size=10,
    )

    logger = L.pytorch.loggers.MLFlowLogger(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        experiment_name="synthetic_forest_only",
        tags={
            "test": "yes",
            "local": "no",
        },
        log_model=True,
    )

    summary = torchinfo.summary(model)
    with tempfile.TemporaryDirectory() as tmp:
        summary_file = f"{tmp}/model_summary.txt"
        with open(summary_file, "w") as f:
            f.write(str(summary))
        logger.experiment.log_artifact(run_id=logger.run_id, local_path=tmp)
        logger.experiment.log_artifact(run_id=logger.run_id, local_path=__file__)

    trainer = L.Trainer(
        max_epochs=250,
        overfit_batches=2,
        logger=logger,
    )
    trainer.fit(
        model=model,
        train_dataloaders=mnist,
    )
