"""A dummy experiment to figure out launching on different platforms."""

import os
import tempfile

import dotenv
import lightning as L
import torch
import torch.nn as nn
import torchinfo
import torchvision


class LitAutoEncoder(L.LightningModule):
    def __init__(self, bottleneck_size: int = 64):
        super().__init__()
        self.bottleneck_size = bottleneck_size

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, self.bottleneck_size),
            nn.ReLU(),
            nn.Linear(self.bottleneck_size, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, self.bottleneck_size),
            nn.ReLU(),
            nn.Linear(self.bottleneck_size, 28 * 28),
        )

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)

        # self.logger is the Lightning wrapper around MLflow,
        # self.logger.experiment is the MlflowClient instance
        self.logger.log_metrics({"loss/train": loss.item()}, step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.logger.log_metrics({"loss/val": loss.item()}, step=self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = torchvision.transforms.ToTensor()

    def prepare_data(self):
        """Prepare the data for setup (download, tokenize, etc.) on one device."""

        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        """Prepare the data for training (split, transform, etc.) on all devices."""

        if stage == "fit":
            full_dataset = torchvision.datasets.MNIST(
                self.data_dir,
                train=True,
                transform=self.transform,
            )
            train_set_size = int(len(full_dataset) * 0.8)
            valid_set_size = len(full_dataset) - train_set_size
            self.train, self.val = torch.utils.data.random_split(
                dataset=full_dataset,
                lengths=[train_set_size, valid_set_size],
                generator=torch.Generator().manual_seed(69),
            )

        if stage == "test":
            raise NotImplementedError()

        if stage == "predict":
            raise NotImplementedError()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        raise NotImplementedError()

    def predict_dataloader(self):
        raise NotImplementedError()


if __name__ == "__main__":
    assert dotenv.load_dotenv(override=True), "Unable to load .env"

    autoencoder = LitAutoEncoder(bottleneck_size=64)
    mnist = MNISTDataModule(
        data_dir="data/raw/mnist",
        batch_size=64,
    )

    logger = L.pytorch.loggers.MLFlowLogger(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),  # Default was set on import
        experiment_name="lightning_logs",
        tags={
            "test": "yes",
            "local": "no",
        },
        log_model=True,
    )

    summary = torchinfo.summary(autoencoder)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = f"{tmp}/model_summary.txt"
        with open(tmp, "w") as f:
            f.write(str(summary))
        logger.experiment.log_artifact(run_id=logger.run_id, local_path=tmp)

    trainer = L.Trainer(
        limit_train_batches=100,
        max_epochs=50,
        overfit_batches=10,
        logger=logger,
    )
    trainer.fit(
        model=autoencoder,
        train_dataloaders=mnist,
    )
