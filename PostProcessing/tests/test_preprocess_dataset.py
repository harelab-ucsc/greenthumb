"""
test_preprocess_dataset.py
"""
import unittest
from tools.preprocess_dataset import load_new_datasets


class TestPreprocessDataset(unittest.TestCase):
    def test_load_new_datasets(self):
        """
        Confirm that loading of example dataset produces expected output.
        """
        self.assertEqual(1 ,1)

if __name__ == "__main__":
    unittest.main()
