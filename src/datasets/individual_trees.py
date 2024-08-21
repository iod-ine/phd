"""A torch_geometric wrapper for the individual trees dataset."""

import functools
import itertools
from typing import Callable, Optional

import kaggle
import torch_geometric


class IndividualTreesBase(torch_geometric.data.InMemoryDataset):
    """Base class for creating datasets based on the individual trees dataset."""

    def __init__(
        self,
        root: str,
        las_features: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        """Create a new instance."""
        self.las_features = las_features
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @functools.cached_property
    def raw_file_names(self):
        """List of files that need to be found in raw_dir to skip the download."""
        return list(
            itertools.chain.from_iterable(
                [
                    [f"Alder/alder_{i:>02}.las" for i in range(26)],
                    [f"Aspen/aspen_{i:>02}.las" for i in range(73)],
                    [f"Birch/birch_{i:>02}.las" for i in range(77)],
                    [f"Fir/fir_{i:>02}.las" for i in range(37)],
                    [f"Pine/pine_{i:>02}.las" for i in range(70)],
                    [f"Spruce/spruce_{i:>02}.las" for i in range(94)],
                    [f"Tilia/tilia_{i:>02}.las" for i in range(17)],
                ],
            )
        )

    @property
    def processed_file_names(self):
        """List of files that need to be found in processed_dir to skip processing."""
        raise NotImplementedError

    def download(self):
        """Download raw data into raw_dir.

        Notes:
            Requires Kaggle API credentials in ~/.kaggle/kaggle.json. For details, see
            https://www.kaggle.com/docs/api#authentication

        """
        kaggle.api.dataset_download_files(
            dataset="sentinel3734/uav-point-clouds-of-individual-trees",
            path=self.raw_dir,
            unzip=True,
        )

    def process(self):
        """Process raw data and save it to processed_dir."""
        raise NotImplementedError
