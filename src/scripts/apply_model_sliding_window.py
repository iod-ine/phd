"""An example script that applies a model in a sliding window."""

import glob

import laspy
import numpy as np
import rasterio
import skimage
import torch
import torch_geometric
import tqdm

import src.clouds
from src.experiments.sf_rgb_mbf_patch import PointNet2TreeSegmentorModule

model_name = "efficient-flea-575"
window_size = 20
height_threshold = 3

if __name__ == "__main__":
    module = PointNet2TreeSegmentorModule.load_from_checkpoint(
        checkpoint_path=f"models/{model_name}.ckpt",
    )

    model = module.pointnet
    model.eval()
    las = laspy.read("data/interim/lysva/als/plot_01.las")

    als_files = sorted(glob.glob("data/interim/lysva/als/plot_*.las"))
    ortho_files = sorted(glob.glob("../data/raw/lysva/ortho/plot_*.tif"))

    transform = torch_geometric.transforms.Compose(
        [
            torch_geometric.transforms.NormalizeScale(),
            torch_geometric.transforms.NormalizeFeatures(),
        ]
    )

    for af, of in zip(als_files, ortho_files):
        las = laspy.read(af)

        with rasterio.open(of) as dd:
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

        xyz = las.xyz[las.xyz[:, 2] >= height_threshold]
        full_pred = torch.empty(xyz.shape[0])
        window_id = torch.empty(xyz.shape[0], dtype=int)
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
                if np.sum(window_mask) < 100:
                    x += window_size
                    continue
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
                    pred = model(data.to("cuda")).cpu().squeeze()
                    full_pred[window_mask] = pred
                    window_id[window_mask] = count

                torch.cuda.empty_cache()

                x += window_size
                count += 1
                pbar.update()

            y += window_size

        pbar.close()

        full_pred = full_pred.numpy()
        window_id = window_id.numpy()

        las = src.clouds.numpy_to_las(xyz=xyz)

        las.add_extra_dim(laspy.ExtraBytesParams(name="pred", type=full_pred.dtype))
        las.add_extra_dim(
            laspy.ExtraBytesParams(name="window_id", type=window_id.dtype)
        )

        las["pred"] = full_pred
        las["window_id"] = window_id

        las.write(f"full_pred_{af.split('/')[-1].split('.')[0]}_{model_name}.laz")
