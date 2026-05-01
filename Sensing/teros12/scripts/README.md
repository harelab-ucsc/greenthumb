# GreenThumb - Sensing - TEROS 12 - Scripts


## Description

This subdirectory contains code related to sensing with TEROS 12 soil sensors.


## Usage

Update sensor_config in teros12_stream.py to match your specific ENTs board serial numbers. Connected ports can be found with `pio device list`, which then list the VID:PID and SER. Copy the SER to sensor config and label the depth.

In your terminal run `python3 teros12_stream` to begin the similtaneous streaming and file logging

CLI Flags: 
    --duration <int> -> run stream for <int> seconds
    --baudrate <int> -> set custom baudrate, default 115200
    --output-folder <folder_path/folder_name> -> where to save csv files

## Authors

jLab

HARE Lab

nubby

dvdthr

## Date

29 Apr 2026


## Version

1.0.0
