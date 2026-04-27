"""
File:
    teros.py

Description:
    High-level interface for working with TEROS-12 soil sensors.

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
import serial
import threading
import time


class Teros12(object):
    """
    Container for drivers for TEROS-12 sensors.
    """
    def __init__(self, port: str, baudrate: int = 115200):
        """
        """
        # Metadata.
        self.baudrate = baudrate
        self.id = "0"
        self.port = port

        # State variables.
        self.connected = False
        self.streaming = False

        # Configure serial channel and streaming.
        self._buffer = []
        self._set_up_serial_connection(baudrate=baudrate, port=port)
        self._set_logging_path()
        self._streaming_thread = None
        self._streaming_stop_event = threading.Event()

    def _set_up_serial_connection(baudrate: int, port: str):
        """
        Description:
            Configure serial connection.

        Args:
            baudrate    (int)
            port        (str)
        """
        # self._conn = Serial connection.
        self._conn = serial.Serial()
        self._conn.baudrate = baudrate
        self._conn.port = port

    def _set_logging_path(self):
        """
        Description:
            Set the logging path for a stream based on current datetime, device
            label, and test ID/label.
        """
        # Generate a string timestamp label.
        ts_dt = datetime.now()
        ts = ts_dt.strftime("%Y%m%d_%H%M%S")

        # Assemble stream path.
        self.log_path = "-".join([
            ts,
            "teros12",
            self.id
        ])

    def _reader(self):
        """
        TODO(nubby, 26 Apr 2026): Make more robust and add writer.
        """
        while not self._streaming_stop_event.is_set():
            line = self._conn.readline()
            if line:
                self._buffer.append(line)

    def connect(self) -> bool:
        """
        Description:
            Initiate a serial connection if possible.

        Returns:
            (bool)  Connected?
        """
        # Open the serial port if not already active.
        if not self.connected:
            self._conn.open()
            self.connected = self._conn.is_open
        return self.connected

    def disconnect(self) -> bool:
        """
        Description:
            Close the serial connection if possible.

        Returns:
            (bool)  Disconnected?
        """
        # Close the serial port if open.
        if self.connected:
            self._conn.close()
            self.connected = self._conn.is_open
        return not self.connected

    def start_stream(self) -> bool:
        """
        Description:
            Initiate streaming if connected and possible.

        Returns:
            (bool)  Streaming?
        """
        # Configure stream logging path.
        self._set_logging_path()
        return self.streaming

    def stop_stream(self) -> bool -> bool -> bool -> bool:
        """
        Description:
            Terminate streaming if connected and possible.

        Returns:
            (bool)  Not streaming?
        """
        return not self.streaming

    def get_metadata(self) -> dict:
        return {}
