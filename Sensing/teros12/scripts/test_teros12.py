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
    @patch("Teros12.datetime")
    def set_up(self, mock_datetime: MagicMock):
        """
        Initialize Teros12 class.
        """
        # Set the initial timestamp to 01/01/01, 01:01:01.
        mock_datetime.now.return_value = datetime(
                2001, 1, 1, 1, 1, 1
            )
        self.teros12 = Teros12()

    def test_connect(self):
        """
        Verify device connections are handled appropriately.
        """
        pass

    @patch("Teros12.datetime")
    def test_disconnect(self):
        """
        Verify that connections are closed and cleaned up appropriately.
        """
        pass

    @patch("Teros12.datetime")
    def test_start_stream(self):
        """
        Test that streaming captures appropriate data.
        """
        pass

    @patch("Teros12.datetime")
    def test_stop_stream(self):
        """
        Test that streaming can be safely stopped.
        """
        pass
    
    @patch("teros12.datetime")
    @patch("teros12.Teros12.connect")
    @patch("teros12.Teros12.disconnect")
    @patch("teros12.Teros12.start_stream")
    @patch("teros12.Teros12.stop_stream")
    def test_get_metadata(self):
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
                    "port": "/dev/tty.TEST"
                    "start_timestamp": "111111_111111",
                    "stop_timestamp": "222222_222222"
                }
            }
        # Set the beginning of streaming timestamp to 11/11/11, 11:11:11.
        mock_datetime.now.return_value = datetime(
                2011, 11, 11, 11, 11, 11
            )
        # Set the ending of streaming timestamp to 22/22/22, 22:22:22.
        mock_datetime.now.return_value = datetime(
                2022, 22, 22, 22, 22, 22
            )

if __name__ == "__main__":
    unittest.main()
