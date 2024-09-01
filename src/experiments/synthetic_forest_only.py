"""Train a PointNet++ tree segmentor directly and only on the synthetic forest data."""

import os
import pathlib
import tempfile
from typing import Optional

import dotenv
import lightning as L
import torch
import torch_geometric
import torch_scatter
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

        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        """Process a single batch of the training dataset and return the loss."""
        pred = self.pointnet(batch)
        loss = nn.functional.mse_loss(pred.squeeze(), batch.y.float())
        per_batch_max_index, _ = torch_scatter.scatter_max(
            src=batch.y,
            index=batch.batch,
        )
        number_of_trees = (per_batch_max_index + 1).sum()
        self.log("loss/train", loss.item() / number_of_trees)

        return loss

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        """Process a single batch of the validation dataset and return the loss."""
        pred = self.pointnet(batch)
        loss = nn.functional.mse_loss(pred.squeeze(), batch.y.float())
        per_batch_max_index, _ = torch_scatter.scatter_max(
            src=batch.y,
            index=batch.batch,
        )
        number_of_trees = (per_batch_max_index + 1).sum()
        self.validation_step_outputs.append(loss / number_of_trees)
        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        """Process the results of the validation epoch."""
        average_loss = torch.stack(self.validation_step_outputs).mean()
        self.log("loss/val", average_loss)

    def configure_optimizers(self):
        """Set up and return the optimizers."""
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=1e-3,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=3,
            gamma=0.5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


class SyntheticForestDataModule(L.LightningDataModule):
    """A data module for the synthetic forest dataset."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        random_seed: int = 42,
        train_samples: int = 100,
        val_samples: int = 20,
        test_samples: int = 20,
        trees_per_sample: int = 50,
        las_features: Optional[list[str]] = None,
    ):
        """Create a new SyntheticForestDataModule instance."""
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = torch_geometric.transforms.Compose(
            [
                torch_geometric.transforms.NormalizeScale(),
                torch_geometric.transforms.NormalizeFeatures(),
            ]
        )
        self.dataset_params = {
            "root": self.data_dir,
            "random_seed": random_seed,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "trees_per_sample": trees_per_sample,
            "las_features": las_features,
        }

    def prepare_data(self):
        """Prepare the data for setup (download, tokenize, etc.) on one device."""
        SyntheticForest(**self.dataset_params)

    def setup(self, stage: str):
        """Prepare the data for training (split, transform, etc.) on all devices."""
        if stage == "fit":
            self.train = SyntheticForest(split="train", **self.dataset_params)
            self.val = SyntheticForest(split="val", **self.dataset_params)

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
    data = SyntheticForestDataModule(
        data_dir="data/raw/synthetic_forest",
        batch_size=2,
    )

    logger = L.pytorch.loggers.MLFlowLogger(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        experiment_name="synthetic_forest_only",
        tags={
            "source": "local",  # local / Kaggle / Colab / DataSphere
        },
        log_model=True,
    )
    logger.experiment.log_artifact(run_id=logger.run_id, local_path=__file__)
    logger.experiment.log_params(run_id=logger.run_id, params=data.dataset_params)

    summary = torchinfo.summary(model)
    with tempfile.TemporaryDirectory() as tmp:
        summary_file = f"{tmp}/model_summary.txt"
        with open(summary_file, "w") as f:
            f.write(str(summary))
        logger.experiment.log_artifact(run_id=logger.run_id, local_path=summary_file)

    trainer = L.Trainer(
        max_epochs=50,
        logger=logger,
        callbacks=[
            L.pytorch.callbacks.EarlyStopping(
                monitor="loss/val",
                mode="min",
                patience=3,
            ),
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="loss/val",
                mode="min",
                dirpath="checkpoints/",
                filename="{epoch}-{loss/val:.2f}",
                save_last=True,
                save_top_k=1,
                every_n_epochs=1,
            ),
        ],
    )
    trainer.fit(
        model=model,
        train_dataloaders=data,
    )
