"""
File:
    test_teros12.py

Description:
    Testing for Teros12 objects.

Authors:
    jLab
    HARE Lab
    nubby

Date:
    26 Apr 2026

Version:
    0.0.1
"""
from datetime import datetime
import unittest
from unittest.mock import patch, MagicMock

import teros12
from teros12 import Teros12


class TestTeros12(unittest.TestCase):
    """
    """
    @patch("teros12.datetime")
    def setUp(self, mock_datetime: MagicMock):
        """
        Initialize Teros12 class.
        """
        # Set the initial timestamp to 01/01/01, 01:01:01.
        mock_datetime.now.return_value = datetime(
                2001, 1, 1, 1, 1, 1
            )
        self.teros12 = Teros12(port="/dev/tty.TEST")

    def test_connect(self):
        """
        Verify device connections are handled appropriately.
        """
        pass

    @patch("teros12.datetime")
    def test_disconnect(self, mock_datetime):
        """
        Verify that connections are closed and cleaned up appropriately.
        """
        pass

    @patch("teros12.datetime")
    def test_start_stream(self, mock_datetime):
        """
        Test that streaming captures appropriate data.
        """
        pass

    @patch("teros12.datetime")
    def test_stop_stream(self, mock_datetime):
        """
        Test that streaming can be safely stopped.
        """
        pass
    
    @patch("teros12.datetime")
    @patch("teros12.Teros12.connect")
    @patch("teros12.Teros12.disconnect")
    @patch("teros12.Teros12.start_stream")
    @patch("teros12.Teros12.stop_stream")
    def test_get_metadata(self, mock_stop, mock_start, mock_disconnect, mock_connect, mock_datetime): # pass all decorators as parameters, reverse order
        """
        Verify that returned metadata is properly formatted.
        """
        expected_metadata = {
                "device": {
                    "type": "TEROS-12",
                    "id": 1
                },
                "depth": {
                    "value": 0.5,
                    "units": "meters"
                },
                "streaming": {
                    "baud_rate": 115200,
                    "port": "/dev/tty.TEST",
                    "start_timestamp": "111111_111111",
                    "stop_timestamp": "222222_222222"
                }
            }

        mock_datetime.now.side_effect = [
            datetime(2011, 11, 11, 11, 11, 11), # set beginning of streaming timestamp to 11/11/11, 11:11:11
            datetime(2022, 2, 22, 22, 22, 22) # Set the ending of streaming timestamp to 22/22/22, 22:22:22.
        ]
        
        self.teros12.start_stream()
        self.teros12.stop_stream()

        metadata = self.teros12.get_metadata()
        self.assertEqual(metadata, expected_metadata)

if __name__ == "__main__":
    unittest.main()
