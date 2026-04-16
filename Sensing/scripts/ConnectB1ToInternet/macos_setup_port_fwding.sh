#!/bin/bash

pfctl -d
pfctl -f ./nat-rules.conf
pfctl -e

sysctl -w net.inet.ip.forwarding=1
#sysctl -w net.inet.ip.fw.enable=1
