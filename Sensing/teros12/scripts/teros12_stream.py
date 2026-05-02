"""
File:
    teros12_stream.py

Description:
    Streaming script for multiple TEROS-12 sensors.

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
import argparse
import time
import json
from serial.tools import list_ports
from teros12 import Teros12

# Define sensor configuration here
sensor_config = [
    {
        "sensor_name" : "sensor_1",
        "depth" : 6,
        "depth_units" : "inch",
        "usb_serial": "aa192e3c9eb2ed11840d4daca7669f5d"
    }
    , 
    {
        "sensor_name" : "sensor_2",
        "depth" : 9,
        "depth_units" : "inch",
        "usb_serial": "14d0f0dda5b2ed118ecd4faca7669f5d"
    },
    {
        "sensor_name" : "sensor_3",
        "depth" : 12,
        "depth_units" : "inch",
        "usb_serial": "9e12fd528fb2ed119ea44eaca7669f5d"
    }
]


def query_for_devices():
    """
    Description: query all connected serial devices and return all usbserial devices

    Returns: a list of all serial devices starting with /dev/cu.usbserial
    """
    devices = []
    for port in list_ports.comports():
        if not port.device.startswith("/dev/cu.usbserial"):
            continue

        device_info = {
            "port": port.device,
            "description": port.description,
            "usb_serial": port.serial_number,
        }
        devices.append(device_info)
    return devices

def print_detected_devices(devices):
    """
    Description: print all the visible devices to the terminal. (not necessary, but was helpful while debugging)
    """
    if not devices:
        print("no serial devices detected")
        return 
    print("Detected serial devices:")
    for index, device in enumerate(devices, start=1):
        print(f"{index}. port={device['port']}")
        print(f"description={device['description']}")
        print(f"usb_serial={device['usb_serial']}")

def assign_device_to_sensors(devices, sensor_config):
    """
    Description:
        Assign serial devices to configured sensors by USB serial number.
        Sensors in sensor_config put have a manually entered serial number
    """
    assignments = []
    
    for sensor in sensor_config:
        if not sensor.get("usb_serial"):
            raise RuntimeError(f"missing usb_serial for {sensor['sensor_name']} in sensor_config")
        
    device_by_serial = {
        device["usb_serial"]: device
        for device in devices
        if device["usb_serial"] is not None
    }

    for sensor in sensor_config:
        usb_serial = sensor["usb_serial"]
        if usb_serial not in device_by_serial:
            raise RuntimeError(f"configured board with serial number {usb_serial} not found")
        
        device = device_by_serial[usb_serial]

        assignments.append({**sensor, "port": device["port"], "usb_serial": device["usb_serial"]})

    return assignments

def create_sensors(assignments, baudrate, output_folder):
    sensors = []
    for assignment in assignments:
        sensor = Teros12(
            port=assignment["port"],
            baudrate=baudrate,
            sensor_name=assignment["sensor_name"],
            depth=assignment["depth"],
            depth_units=assignment["depth_units"],
            usb_serial=assignment["usb_serial"],
            output_folder=output_folder,
            )
        sensors.append(sensor)
    return sensors

def start_all_streams(sensors):
    for sensor in sensors:
        print(f"starting {sensor.sensor_name} on {sensor.port}...")
        startStream = sensor.start_stream()
        print(f"started={startStream}")

def stop_all_streams(sensors):
    for sensor in sensors:
        print(f"stopping {sensor.sensor_name}...")
        sensor.stop_stream()
        sensor.disconnect()
        print(f"metadata={sensor.get_metadata()}")
        

def teros12_stream():
    """
    """
    parser = argparse.ArgumentParser(description="stream from multiple teros12 sensors")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--output-folder", default="output")
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()


    devices = query_for_devices()
    print_detected_devices(devices)

    assignments = assign_device_to_sensors(devices, sensor_config)
    for assignment in assignments:
        print(
            f"{assignment['sensor_name']}"
            f"depth={assignment['depth']}{assignment['depth_units']}"
            f"port={assignment['port']}"
            f"usb_serial={assignment['usb_serial']}"
            )

    sensors = create_sensors(assignments=assignments, baudrate=args.baudrate,output_folder=args.output_folder)

    try:
        start_all_streams(sensors)
        if args.duration is not None:
            time.sleep(args.duration)
        else:
            print("streaming indefinitely. ctrl c to stop")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("stopping all streams")
    finally:
        stop_all_streams(sensors)

if __name__ == "__main__":
    teros12_stream()
