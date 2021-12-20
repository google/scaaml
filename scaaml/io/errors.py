"""Implement an error to indicate that a scaaml.io.Dataset already exists.

Creating scaaml.io.Dataset should not overwrite existing files. When it could
the constructor needs to raise an error, which should also contain the dataset
directory.
"""

from pathlib import Path


class DatasetExistsError(FileExistsError):
    """Error for signalling that the dataset already exists."""
    def __init__(self, dataset_path: Path) -> None:
        """Represents that the dataset already exists.

        Args:
          dataset_path: The dataset path.
        """
        super().__init__(
            f'Dataset info file exists and would be overwritten. Use instead:'
            f' Dataset.from_config(dataset_path="{dataset_path}")')
        self.dataset_path = dataset_path
