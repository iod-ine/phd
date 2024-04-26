"""A dummy experiment to figure out launching on different platforms."""

import lightning as L
import torch
import torch.nn as nn
import torchvision


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("loss/train", loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("loss/val", loss.detach(), prog_bar=True)

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
    encoder = nn.Sequential(
        nn.Linear(28 * 28, 64),
        nn.ReLU(),
        nn.Linear(64, 3),
    )

    decoder = nn.Sequential(
        nn.Linear(3, 64),
        nn.ReLU(),
        nn.Linear(64, 28 * 28),
    )

    autoencoder = LitAutoEncoder(encoder, decoder)
    mnist = MNISTDataModule(
        data_dir="data/raw/mnist",
        batch_size=64,
    )

    trainer = L.Trainer(
        # limit_train_batches=100,
        max_epochs=100,
        overfit_batches=10,
    )
    trainer.fit(
        model=autoencoder,
        train_dataloaders=mnist,
    )
