"""An example script that applies a model in a sliding window."""

import laspy
import numpy as np
import rasterio
import skimage
import torch
import torch_geometric
import tqdm

from src.experiments.sf_rgb_mbf_patch import PointNet2TreeSegmentorModule

if __name__ == "__main__":
    module = PointNet2TreeSegmentorModule.load_from_checkpoint(
        "models/monumental-lark-777.ckpt"
    )

    model = module.pointnet
    model.eval()
    las = laspy.read("data/interim/lysva/als/plot_01.las")

    with rasterio.open("data/raw/lysva/ortho/plot_01.tif") as dd:
        ortho = dd.read()

    multiscale_features = skimage.feature.multiscale_basic_features(
        image=ortho,
        channel_axis=0,
        intensity=True,
        edges=True,
        texture=True,
        sigma_min=0.5,
        sigma_max=8,
        num_sigma=None,
    )

    transformer = rasterio.transform.AffineTransformer(dd.transform)
    _, height, width = ortho.shape
    transform = torch_geometric.transforms.Compose(
        [
            torch_geometric.transforms.NormalizeScale(),
            torch_geometric.transforms.NormalizeFeatures(),
        ]
    )

    window_size = 12
    height_threshold = 2
    xyz = las.xyz[las.xyz[:, 2] >= height_threshold]
    full_pred = torch.empty(xyz.shape[0])
    min_x, min_y, _ = xyz.min(axis=0)
    max_x, max_y, _ = xyz.max(axis=0)

    count = 0
    pbar = tqdm.tqdm()

    x, y = min_x, min_y

    while y < max_y:
        x = min_x
        while x < max_x:
            window_mask = (
                (xyz[:, 0] >= x)
                & (xyz[:, 0] < x + window_size)
                & (xyz[:, 1] >= y)
                & (xyz[:, 1] < y + window_size)
            )
            pos = xyz[window_mask]

            row, col = transformer.rowcol(pos[:, 0], pos[:, 1])
            row = [x if x < height else height - 1 for x in row]
            col = [x if x < width else width - 1 for x in col]

            rgb = ortho[:, row, col].astype(np.float32)
            rgb = np.rollaxis(rgb, axis=1)
            mbf = multiscale_features[row, col, :]
            features = np.hstack([rgb, mbf])

            data = transform(
                torch_geometric.data.Data(
                    pos=torch.from_numpy(pos.astype(np.float32)),
                    x=torch.from_numpy(features),
                    batch=torch.zeros(pos.shape[0], dtype=int),
                )
            )

            with torch.no_grad():
                full_pred[window_mask] = model(data).squeeze() + count

            x += window_size
            count += 1
            pbar.update()

        y += window_size

    pbar.close()
