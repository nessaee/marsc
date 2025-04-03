#!/usr/bin/env python3

import sys
import os
import time
import logging
import datetime

# Import modules from our package
from marsc.camera import MarsCamera
from marsc.image_processor import ImageProcessor
from marsc.ui.camera_viewer import CameraViewer
from marsc.utils.arg_parser import parse_arguments

def setup_logging(debug=False):
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"marsc_{timestamp}.log")
    
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create a logger for the main module
    logger = logging.getLogger('marsc.main')
    logger.info(f"Logging initialized at {log_level}")
    logger.info(f"Log file: {log_file}")
    
    return logger

def main():
    """Main entry point for Mars Camera application"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(debug=args.debug)
    
    try:
        # List available cameras
        logger.info("Searching for cameras...")
        cameras = MarsCamera.list_cameras()
        if not cameras:
            logger.error("No Player One cameras found")
            print("No Player One cameras found")
            return
        
        logger.info(f"Found {len(cameras)} camera(s)")
        print(f"Found {len(cameras)} camera(s):")
        for i, cam in enumerate(cameras):
            logger.info(f"Camera {i}: {cam['name']} (S/N: {cam['sn']})")
            print(f"[{i}] {cam['name']} (S/N: {cam['sn']})")
        
        # Connect to camera
        logger.info(f"Connecting to camera: SN={args.sn if args.sn else 'None'}, index={args.index}")
        camera = MarsCamera(serial_number=args.sn, index=args.index)
        if not camera.connect():
            logger.error("Failed to connect to camera")
            print("Failed to connect to camera")
            return
        
        logger.info(f"Connected to {camera.get_name()} (S/N: {camera.get_sn()})")
        print(f"Connected to {camera.get_name()} (S/N: {camera.get_sn()})")
        
        # Get camera size (fixed at 1944x1096)
        width, height = camera.get_size()
        logger.info(f"Camera resolution: {width}x{height}")
        print(f"Resolution: {width}x{height}")
        
        logger.debug(f"Setting ROI to full frame: (0,0,{width},{height})")
        camera.set_roi(0, 0, width, height)
        
        # Create image processor
        logger.info("Initializing image processor")
        processor = ImageProcessor()
        
        # Set raw mode based on arguments
        if hasattr(args, 'monochrome') and args.monochrome:
            logger.info("Setting camera to monochrome mode")
            processor.set_raw_mode(True)
            print("Camera set to monochrome mode")
        
        # Create and start camera viewer
        logger.info("Creating camera viewer")
        viewer = CameraViewer(camera, processor)
        
        # Initialize with command line settings
        logger.info(f"Initial settings: exposure={args.exposure}ms, gain={args.gain}, offset={args.offset}")
        initial_settings = {
            'exposure': args.exposure,
            'gain': args.gain,
            'offset': args.offset,
            # Fixed resolution of 1944x1096 - no binning needed
            'usb_limit': args.usb_limit,
            'cooler_temp': args.cooler
        }
        
        logger.info("Starting camera viewer")
        if viewer.start(initial_settings):
            # Run the viewer main loop
            logger.info(f"Running main loop with save_dir={args.save_dir if args.save_dir else 'None'}")
            viewer.run(save_dir=args.save_dir)
    
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
        print("\nProgram terminated by user")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        print(f"\nError: {e}")
    finally:
        # Clean up
        if 'camera' in locals() and camera.connected:
            logger.info("Disconnecting camera")
            print("Disconnecting camera...")
            camera.disconnect()
        logger.info("Application shutdown complete")

if __name__ == "__main__":
    main()
