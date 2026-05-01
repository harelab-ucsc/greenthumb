"""
File:
    teros.py

Description:
    High-level interface for working with TEROS-12 soil sensors.

Authors:
    jLab
    HARE Lab
    nubby
    dvdthr

Date:
    26 Apr 2026

Version:
    0.0.1
"""
from datetime import datetime
import serial
import threading
import time
import os


class Teros12(object):
    """
        Descrciption: 
            Container for one teros12 sensor
        arguments:
            port: seial port path
            baudrate: default ENTs board baudrate
            sensor_name: sensor label
            depth: self explanatory
            depth_units: default to centimeters
            usb_serial: unique serial number attached to each board
            output_folder: where to write output data to
    """
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        sensor_name: str = "sensor_0",
        depth: float | None = None, 
        depth_units = "inch",
        usb_serial: str | None = None,
        output_folder: str = "output"):
       
        # Metadata.
        self.baudrate = baudrate
        self.sensor_name = sensor_name
        self.port = port
        self.depth = depth
        self.depth_units = depth_units
        self.usb_serial = usb_serial
        self.output_folder = output_folder
        self.start_timestamp = None
        self.stop_timestamp = None


        # State variables.
        self.connected = False
        self.streaming = False

        # Configure serial channel and streaming.
        self._buffer = []
        self._set_up_serial_connection(baudrate=baudrate, port=port)
        self.log_path = None
        self._streaming_thread = None
        self._streaming_stop_event = threading.Event()

    def _set_up_serial_connection(self, baudrate: int, port: str):
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
        self._conn.timeout = 1

    def _set_logging_path(self):
        """
        Description:
            Set the logging path for a stream based on current datetime, device
            label, and test ID/label.
        """
        # Generate a string timestamp label.
        ts_dt = datetime.now()
        ts = ts_dt.strftime("%Y%m%d_%H%M%S")

        # Assemble csv filename

        depth_label = "unknown_depth"
        if self.depth is not None: 
            depth_label = f"{self.depth}{self.depth_units}"

        filename = "-".join([ 
            depth_label,
            "teros12",
            ts,
            self.sensor_name
        ]) + ".csv" 
        os.makedirs(self.output_folder, exist_ok=True) # make sure output directory exists
        self.log_path = os.path.join(self.output_folder, filename) # save log path

    def _reader(self):
        """
        TODO(nubby, 26 Apr 2026): Make more robust and add writer.
        """
        while not self._streaming_stop_event.is_set(): 
            raw_line = self._conn.readline()

            if not raw_line: #if no incoming data, wait
                continue

            line = raw_line.decode("utf-8", errors="replace").strip() 

            if not line: 
                continue

            self._buffer.append(line)

            vwc = self._parse_vwc(line)

            if vwc is None:
                continue
                
            timestamp = datetime.now().isoformat(timespec="seconds")
            self._write_csv_row(timestamp=timestamp, vwc=vwc, raw_line=line)
            
    def _parse_vwc(self, line: str) -> float | None: 
        """
        Description: parse volumetric water content value from serial output
        """
        key = "vwcPercentMineralizedSoils:"
        if key not in line: 
            return None
        value_text = line.split(key, 1)[1].strip()
        value_text = value_text.split(";", 1)[0].strip()

        try: 
            return float(value_text)
        except ValueError: 
            return None
        
    def _write_csv_row(self, timestamp: str, vwc: float, raw_line: str):
        """
        Description: write the timestamp, sensor name, sensor depth, parsed vwc, and raw line to the csv log file
        """
        if self.log_path is None:
            self._set_logging_path()
        file_exists = os.path.exists(self.log_path)
        file_empty = not file_exists or os.path.getsize(self.log_path) == 0

        with open(self.log_path, "a", encoding="utf-8") as log_file: 
            if file_empty:
                # write description if file is empty
                log_file.write(f"BOARD SERIAL NUMBER: {self.usb_serial}\nUSB SERIAL PORT: {self.port}\n")
                log_file.write("timestamp,sensor_name,depth,depth_units,VWC,raw_line\n") 
            safe_raw_line = raw_line.replace('"','""')
            log_file.write(f'{timestamp},{self.sensor_name},{self.depth},{self.depth_units},{vwc},"{safe_raw_line}"\n')

    def connect(self) -> bool:
        """
        Description:
            Initiate a serial connection if possible.

        Returns:
            (bool)  Connected?
        """
        # Open the serial port if not already active.
        if not self.connected:
            try:
                self._conn.open()
            except serial.SerialException:
                return False
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
        if self.streaming:
            self.stop_stream()

        if self.connected:
            try:
                self._conn.close()
            except serial.SerialException:
                return False
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
        if self.streaming:
            return True
        if not self.connected:
            if not self.connect():
                return False
        self._set_logging_path()
        self.start_timestamp = datetime.now().isoformat(timespec="seconds")
        self.stop_timestamp = None

        self._streaming_stop_event.clear()
        self.streaming = True

        self._streaming_thread = threading.Thread(target=self._reader, daemon=True)
        self._streaming_thread.start()

        return self.streaming

    def stop_stream(self) -> bool:
        """
        Description:
            Terminate streaming if connected and possible.

        Returns:
            (bool)  Not streaming?
        """

        if not self.streaming:
            return True
        
        self._streaming_stop_event.set()
        
        if self._streaming_thread is not None:
            self._streaming_thread.join(timeout=3)
        
        self.streaming = False
        self.stop_timestamp = datetime.now().isoformat(timespec="seconds")

        return not self.streaming

    def get_metadata(self) -> dict:
        """
        Description: 
            Get and return all metadata
        """
        res = {}
        res["sensor_name"] = self.sensor_name
        res["port"] = self.port
        res["depth"] = self.depth
        res["depth_units"] = self.depth_units
        res["usb_serial"] = self.usb_serial
        res["output_folder"] = self.output_folder
        res["start_timestamp"] = self.start_timestamp
        res["stop_timestamp"] = self.stop_timestamp
        res["connected"] = self.connected
        res["streaming"] = self.streaming
        res["baudrate"] = self.baudrate
        res["log_path"] = self.log_path

        return res
