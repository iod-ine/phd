"""Train a PointNet++ tree segmentor directly and only on the synthetic forest data."""

import os
import tempfile
from typing import Callable, Optional

import dotenv
import laspy
import lightning as L
import mlflow
import numpy as np
import torch
import torch_geometric
import torch_scatter
import torchinfo
from torch import nn

import src.clouds
import src.visualization.clouds
from src.datasets import SyntheticForestRGBGrid
from src.metrics import Accuracy
from src.models.pointnet import PointNet2TreeSegmentor
from src.transforms import PerTreeRandomRotateScale


class PointNet2TreeSegmentorModule(L.LightningModule):
    """A PointNet++ tree segmentor lightning module."""

    def __init__(
        self,
        loss: Optional[Callable] = None,
        lr: float = 1e-2,
        lr_start_factor: float = 0.1,
        lr_warmup_iters: int = 3,
        lr_decay_iters: int = 20,
        lr_end_factor: float = 0.1,
    ):
        """Create a new LitPointNet2TreeSegmentor instance."""
        super().__init__()

        self.pointnet = PointNet2TreeSegmentor(num_features=3)
        self.accuracy = Accuracy()
        self.loss = loss or nn.MSELoss()

        self.lr = lr
        self.lr_start_factor = lr_start_factor
        self.lr_warmup_iters = lr_warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.lr_end_factor = lr_end_factor

        self.save_hyperparameters(ignore=["loss"])

        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        """Process a single batch of the training dataset and return the loss."""
        pred = self.pointnet(batch)
        loss = self.loss(pred.squeeze(), batch.y.float())
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
        loss = self.loss(pred.squeeze(), batch.y.float())
        self.validation_step_outputs.append(loss)
        self.accuracy(pred.squeeze().round(), batch.y)

    def on_validation_epoch_end(self):
        """Process the results of the validation epoch."""
        average_loss = torch.stack(self.validation_step_outputs).mean()
        self.log("loss/val", average_loss)
        self.log("accuracy/val", self.accuracy.compute())
        self.validation_step_outputs.clear()
        self.accuracy.reset()

    def configure_optimizers(self):
        """Set up and return the optimizers."""
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer=optimizer,
                    start_factor=self.lr_start_factor,
                    end_factor=1.0,
                    total_iters=self.lr_warmup_iters,
                ),
                torch.optim.lr_scheduler.LinearLR(
                    optimizer=optimizer,
                    start_factor=1.0,
                    end_factor=self.lr_end_factor,
                    total_iters=self.lr_decay_iters,
                ),
            ],
            milestones=[self.lr_warmup_iters],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


class SyntheticForestColoredDataModule(L.LightningDataModule):
    """A data module for the synthetic forest dataset."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        random_jitter: float = 0.2,
        random_scale_range: tuple[float, float] = (0.9, 1.1),
        random_rotate_degrees_range: tuple[float, float] = (-180, 180),
        random_seed: int = 42,
        train_samples: int = 100,
        val_samples: int = 20,
        trees_per_sample: int = 50,
        height_threshold: float = 2.0,
        n_cols: Optional[int] = None,
        dx: float = 5.0,
        dy: float = 5.0,
        xy_noise_mean: float = 0.0,
        xy_noise_std: float = 1.0,
        num_workers: int = 11,
    ):
        """Create a new data module instance."""
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.random_jitter = random_jitter
        self.random_scale_range = random_scale_range
        self.random_rotate_degrees_range = random_rotate_degrees_range
        self.transform = torch_geometric.transforms.Compose(
            [
                torch_geometric.transforms.RandomJitter(translate=random_jitter),
                PerTreeRandomRotateScale(
                    scales=random_scale_range,
                    degrees=random_rotate_degrees_range,
                ),
                torch_geometric.transforms.NormalizeScale(),
                torch_geometric.transforms.NormalizeFeatures(),
            ]
        )
        self.val_transform = torch_geometric.transforms.Compose(
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
            "trees_per_sample": trees_per_sample,
            "height_threshold": height_threshold,
            "n_cols": n_cols,
            "dx": dx,
            "dy": dy,
            "xy_noise_mean": xy_noise_mean,
            "xy_noise_std": xy_noise_std,
        }
        self.num_workers = num_workers

    def prepare_data(self):
        """Prepare the data for setup (download, tokenize, etc.) on one device."""
        SyntheticForestRGBGrid(**self.dataset_params)

    def setup(self, stage: str):
        """Prepare the data for training (split, transform, etc.) on all devices."""
        if stage == "fit":
            self.train = SyntheticForestRGBGrid(
                split="train",
                **self.dataset_params,
                transform=self.transform,
            )
            self.val = SyntheticForestRGBGrid(
                split="val",
                **self.dataset_params,
                transform=self.val_transform,
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
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Set up and return the validation data loader."""
        return torch_geometric.loader.DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Set up and return the test data loader."""
        raise NotImplementedError()

    def predict_dataloader(self):
        """Set up and return the prediction data loader."""
        raise NotImplementedError()


if __name__ == "__main__":
    assert dotenv.load_dotenv(override=True)

    model = PointNet2TreeSegmentorModule()
    data = SyntheticForestColoredDataModule(
        data_dir="data/raw/synthetic_forest",
        batch_size=1,
        trees_per_sample=36,
        random_seed=91,
        dx=4,
        dy=4,
        xy_noise_std=2,
    )

    logger = L.pytorch.loggers.MLFlowLogger(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        experiment_name="synthetic_forest_only",
        tags={
            "environment": "local",
        },
        log_model=True,
    )

    mlflow.log_artifact(run_id=logger.run_id, local_path=__file__)
    mlflow.log_params(run_id=logger.run_id, params=data.dataset_params)

    summary = torchinfo.summary(model)
    with tempfile.TemporaryDirectory() as tmp:
        summary_file = f"{tmp}/model_summary.txt"
        with open(summary_file, "w") as f:
            f.write(str(summary))
        mlflow.log_artifact(run_id=logger.run_id, local_path=summary_file)

    trainer = L.Trainer(
        fast_dev_run=True,
        max_epochs=50,
        accelerator="cpu",
        logger=logger,
        log_every_n_steps=5,
        accumulate_grad_batches=5,
        enable_progress_bar=True,
        callbacks=[
            L.pytorch.callbacks.EarlyStopping(
                monitor="loss/val",
                mode="min",
                patience=5,
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
            L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ],
    )
    trainer.fit(
        model=model,
        train_dataloaders=data,
    )

    example = data.val[0]
    example["batch"] = torch.zeros_like(example.y)

    model = model.pointnet.to("cpu")
    model.eval()
    with torch.no_grad():
        pred = model(example)

    las = src.clouds.pyg_data_to_las(example)
    las.add_extra_dim(laspy.ExtraBytesParams(name="pred", type=np.float32))
    las["pred"][:] = pred.cpu().squeeze().numpy()
    las.write("example_prediction.laz")
    mlflow.log_artifact(run_id=logger.run_id, local_path="example_prediction.laz")

    ax = src.visualization.clouds.scatter_point_cloud_3d(
        las.xyz,
        color=las["pred"],
    )
    logger.experiment.log_figure(
        run_id=logger.run_id,
        figure=ax.figure,
        artifact_file="example_prediction.png",
    )
