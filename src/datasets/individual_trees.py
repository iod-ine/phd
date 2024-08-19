"""A torch_geometric wrapper for the individual trees dataset."""

from torch_geometric.data import InMemoryDataset


class IndividualTreesDataset(InMemoryDataset):
    """"""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        """Create a new IndividualTreesDataset instance."""
        super().__init__(root, transform, pre_transform, pre_filter)

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

    @functools.cached_property
    def processed_file_names(self):
        """List of files that need to be found in processed_dir to skip processing."""
        pass

    def download(self):
        """Download raw data into raw_dir."""
        kaggle.api.dataset_download_files(
            dataset="sentinel3734/uav-point-clouds-of-individual-trees",
            path=self.raw_dir,
            unzip=True,
        )

    def process(self):
        """Process raw data and save it to processed_dir."""
        pass
