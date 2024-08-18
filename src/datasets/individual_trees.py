"""A torch_geometric wrapper for the individual trees dataset."""

from torch_geometric.data import InMemoryDataset


class IndividualTreesDataset(InMemoryDataset):
    """"""

    def __init__(self):
        """Create a new IndividualTreesDataset instance."""
        super().__init__()

    @property
    def raw_file_names(self):
        """List of files that need to be found in raw_dir to skip the download."""
        pass

    @property
    def processed_file_names(self):
        """List of files that need to be found in processed_dir to skip processing."""
        pass

    def download(self):
        """Download raw data into raw_dir."""
        pass

    def process(self):
        """Process raw data and save it to processed_dir."""
        pass
