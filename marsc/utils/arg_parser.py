#!/usr/bin/env python3

import argparse

def parse_arguments():
    """Parse command line arguments for Mars Camera Viewer"""
    parser = argparse.ArgumentParser(description='Mars Camera Viewer')
    parser.add_argument('--sn', help='Camera serial number')
    parser.add_argument('--index', type=int, default=0, help='Camera index (0-based)')
    parser.add_argument('--exposure', type=float, default=10.0, help='Exposure time (ms)')
    parser.add_argument('--gain', type=float, default=0, help='Gain value')
    parser.add_argument('--offset', type=int, default=10, help='Black level offset')
    # Fixed resolution of 1944x1096 - no binning option needed
    parser.add_argument('--usb-limit', type=int, default=80, help='USB bandwidth limit (35-100%)')
    parser.add_argument('--cooler', type=int, default=-10, help='Cooler target temperature (C)')
    parser.add_argument('--save-dir', default='.', help='Directory for saving images')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--monochrome', action='store_true', help='Treat sensor as monochrome (no debayering)')
    
    return parser.parse_args()
