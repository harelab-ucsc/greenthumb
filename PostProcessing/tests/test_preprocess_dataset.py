"""
test_preprocess_dataset.py
"""
import unittest
from tools.preprocess_dataset import label_datasets

import pandas as pd


# Global variables.

## Example dataset paths.

### Grount truth labels.
PATH_EXAMPLE_CORE_LABELS = "tests/examples/core_labels.csv"
PATH_EXAMPLE_PEN_LABELS = "tests/examples/pen_labels.csv"

### Dataset directory.
PATH_EXAMPLE_DATASET_DIR_INPUT = "tests/examples"

### Fully-labeled datasets.
PATH_EXAMPLE_DATASET_DIR_PROCESSED = "tests/examples/processed"
PATH_EXAMPLE_DATASET_LABELED_BASIC = ("tests/examples/processed/"
                                      "19700101_000000-lab-c0-w0.csv")


## Helper functions.
def _load_csv_as_dataset(csv_path) -> pd.DataSet:
    """Read CSV file and return a dataset."""
    return pd.read_csv(csv_path)

class TestPreprocessDataset(unittest.TestCase):
    def test_label_datasets(self):
        """
        Confirm that labeling produces expected output datasets.
        """
        example_core_labels = _load_csv_as_dataset(PATH_EXAMPLE_CORE_LABELS)
        example_pen_labels = _load_csv_as_dataset(PATH_EXAMPLE_PEN_LABELS)
        example_labeled_dataset = _load_csv_as_dataset(
                PATH_EXAMPLE_DATASET_LABELED_BASIC
            )
        processed_dataset = label_dataset(
                core_labels=example_core_labels,
                pen_labels=example_pen_labels,
                data_in_dir=PATH_EXAMPLE_DATASET_DIR_INPUT
            )
        self.assertEqual(processed_dataset, example_labeled_dataset)


if __name__ == "__main__":
    unittest.main()
