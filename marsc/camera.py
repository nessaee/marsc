#!/usr/bin/env python3

import os
import sys

# Ensure the pyPOACamera module is available
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)

# Import existing pyPOACamera module
try:
    import numpy as np
    import time
    from pyPOACamera import *
except ImportError:
    print("Error: pyPOACamera.py module not found.")
    print("Make sure pyPOACamera.py is in the same directory as this script.")
    sys.exit(1)

class MarsCamera:
    """Wrapper for PlayerOne Mars cameras with enhanced features"""
    
    def __init__(self, serial_number=None, index=0):
        self.camera = None
        self.sn = serial_number
        self.index = index
        self.connected = False
        self.streaming = False
        self.cooler_on = False
    
    @staticmethod
    def list_cameras():
        """Get list of available cameras"""
        cameras = []
        count = GetCameraCount()
        
        for i in range(count):
            status, props = GetCameraProperties(i)
            if status == POAErrors.POA_OK:
                cameras.append({
                    'name': props.cameraModelName.decode('utf-8').strip('\0'),
                    'sn': props.SN.decode('utf-8').strip('\0'),
                    'id': props.cameraID,
                    'is_color': bool(props.isColorCamera),
                    'has_cooler': bool(props.isHasCooler),
                    'bit_depth': props.bitDepth,
                    'max_width': props.maxWidth,
                    'max_height': props.maxHeight,
                    'bayer_pattern': props.bayerPattern_
                })
        
        return cameras
    
    def connect(self):
        """Connect to camera and configure for Sony IMX462 sensor"""
        cameras = self.list_cameras()
        if not cameras:
            print("No cameras found")
            return False
        
        # Find camera by serial number or index
        if self.sn:
            for cam in cameras:
                if cam['sn'] == self.sn:
                    self.camera_id = cam['id']
                    self.info = cam
                    break
            if not hasattr(self, 'camera_id'):
                print(f"Camera with S/N {self.sn} not found")
                return False
        else:
            if self.index >= len(cameras):
                self.index = 0
            self.camera_id = cameras[self.index]['id']
            self.info = cameras[self.index]
            self.sn = cameras[self.index]['sn']
        
        # Open camera
        status = OpenCamera(self.camera_id)
        if status != POAErrors.POA_OK:
            print(f"Error opening camera: {GetErrorString(status)}")
            return False
        
        # Initialize camera
        status = InitCamera(self.camera_id)
        if status != POAErrors.POA_OK:
            print(f"Error initializing camera: {GetErrorString(status)}")
            CloseCamera(self.camera_id)
            return False
        
        self.connected = True
        print(f"Connected to {self.get_name()} (S/N: {self.get_sn()})")
        
        # Set fixed resolution of 1944x1096 (Sony IMX462 max resolution)
        self.set_size(1944, 1096)
        
        # Configure optimal settings for Sony IMX462 sensor
        # Set image format to RAW16 for 12-bit ADC (as per sensor specs)
        status = SetImageFormat(self.camera_id, POAImgFormat.POA_RAW16)
        if status != POAErrors.POA_OK:
            print(f"Warning: Could not set image format: {GetErrorString(status)}")
            
        # Set USB bandwidth limit to 90% for optimal performance with USB 3.0
        status = SetConfig(self.camera_id, POAConfig.POA_USB_BANDWIDTH_LIMIT, 90, False)
        if status != POAErrors.POA_OK:
            print(f"Warning: Could not set USB bandwidth limit: {GetErrorString(status)}")
            
        # Disable pixel binning sum (since we're using fixed resolution)
        status = SetConfig(self.camera_id, POAConfig.POA_PIXEL_BIN_SUM, 0, False)
        if status != POAErrors.POA_OK:
            print(f"Warning: Could not disable pixel binning: {GetErrorString(status)}")
        
        print(f"Camera initialized at 1944x1096 resolution with 12-bit depth")
        return True
    
    def disconnect(self):
        """Disconnect from camera"""
        if not self.connected:
            return
        
        if self.streaming:
            self.stop_streaming()
        
        if self.cooler_on:
            self.cooler_off()
        
        CloseCamera(self.camera_id)
        self.connected = False
    
    def get_name(self):
        """Get camera name"""
        return self.info['name'] if self.info else "Unknown"
    
    def get_sn(self):
        """Get camera serial number"""
        return self.sn
    
    def get_bit_depth(self):
        """Get bit depth"""
        return self.info['bit_depth'] if self.info else 0
    
    def has_cooler(self):
        """Check if camera has cooler"""
        return self.info['has_cooler'] if self.info else False
    
    def is_color(self):
        """Check if camera is color"""
        return self.info['is_color'] if self.info else False
    
    def get_size(self):
        """Get current image dimensions"""
        if not self.connected:
            return (0, 0)
        
        status, width, height = GetImageSize(self.camera_id)
        if status != POAErrors.POA_OK:
            print(f"Error getting image size: {GetErrorString(status)}")
            return (0, 0)
        
        return (width, height)
    
    def set_roi(self, x, y, width, height):
        """Set region of interest"""
        if not self.connected:
            return False
        
        was_streaming = self.streaming
        if was_streaming:
            self.stop_streaming()
        
        status = SetImageStartPos(self.camera_id, x, y)
        if status != POAErrors.POA_OK:
            print(f"Error setting start position: {GetErrorString(status)}")
            if was_streaming:
                self.start_streaming()
            return False
        
        status = SetImageSize(self.camera_id, width, height)
        if status != POAErrors.POA_OK:
            print(f"Error setting image size: {GetErrorString(status)}")
            if was_streaming:
                self.start_streaming()
            return False
        
        if was_streaming:
            self.start_streaming()
        
        return True
    
    # Removed set_binning method as we now use fixed resolution of 1944x1096
    
    def set_size(self, width, height):
        """Set image size"""
        if not self.connected:
            return False
            
        was_streaming = self.streaming
        if was_streaming:
            self.stop_streaming()
            
        status = SetImageSize(self.camera_id, width, height)
        if status != POAErrors.POA_OK:
            print(f"Error setting image size: {GetErrorString(status)}")
            if was_streaming:
                self.start_streaming()
            return False
            
        if was_streaming:
            self.start_streaming()
            
        return True
    
    def set_exposure(self, exposure_ms):
        """Set exposure in milliseconds"""
        if not self.connected:
            return False
        
        exposure_us = int(exposure_ms * 1000)
        status = SetExp(self.camera_id, exposure_us, False)
        if status != POAErrors.POA_OK:
            print(f"Error setting exposure: {GetErrorString(status)}")
            return False
        
        return True
    
    def get_exposure(self):
        """Get exposure in milliseconds"""
        if not self.connected:
            return 0
        
        status, exp_us, is_auto = GetExp(self.camera_id)
        if status != POAErrors.POA_OK:
            print(f"Error getting exposure: {GetErrorString(status)}")
            return 0
        
        return exp_us / 1000.0
    
    def set_gain(self, gain):
        """Set gain"""
        if not self.connected:
            return False
        
        status = SetGain(self.camera_id, int(gain), False)
        if status != POAErrors.POA_OK:
            print(f"Error setting gain: {GetErrorString(status)}")
            return False
        
        return True
    
    def get_gain(self):
        """Get gain"""
        if not self.connected:
            return 0
        
        status, gain, is_auto = GetGain(self.camera_id)
        if status != POAErrors.POA_OK:
            print(f"Error getting gain: {GetErrorString(status)}")
            return 0
        
        return gain
    
    def set_offset(self, offset):
        """Set offset"""
        if not self.connected:
            return False
        
        status = SetConfig(self.camera_id, POAConfig.POA_OFFSET, offset, False)
        if status != POAErrors.POA_OK:
            print(f"Error setting offset: {GetErrorString(status)}")
            return False
        
        return True
    
    def set_usb_limit(self, limit_percent):
        """Set USB bandwidth limit (35-100%)"""
        if not self.connected:
            return False
        
        limit = max(35, min(100, limit_percent))
        status = SetConfig(self.camera_id, POAConfig.POA_USB_BANDWIDTH_LIMIT, limit, False)
        if status != POAErrors.POA_OK:
            print(f"Error setting USB bandwidth limit: {GetErrorString(status)}")
            return False
        
        return True
    
    def set_target_temp(self, temp):
        """Set target temperature"""
        if not self.connected or not self.has_cooler():
            return False
        
        status = SetConfig(self.camera_id, POAConfig.POA_TARGET_TEMP, temp, False)
        if status != POAErrors.POA_OK:
            print(f"Error setting target temperature: {GetErrorString(status)}")
            return False
        
        return True
    
    def get_temp(self):
        """Get current temperature"""
        if not self.connected or not self.has_cooler():
            return None
        
        status, temp = GetCameraTEMP(self.camera_id)
        if status != POAErrors.POA_OK:
            print(f"Error getting temperature: {GetErrorString(status)}")
            return None
        
        return temp
    
    def cooler_on(self):
        """Turn cooler on"""
        if not self.connected or not self.has_cooler():
            return False
        
        status = SetConfig(self.camera_id, POAConfig.POA_COOLER, 1, 0)
        if status != POAErrors.POA_OK:
            print(f"Error turning cooler on: {GetErrorString(status)}")
            return False
        
        self.cooler_on = True
        return True
    
    def cooler_off(self):
        """Turn cooler off"""
        if not self.connected or not self.has_cooler() or not self.cooler_on:
            return False
        
        status = SetConfig(self.camera_id, POAConfig.POA_COOLER, 0, 0)
        if status != POAErrors.POA_OK:
            print(f"Error turning cooler off: {GetErrorString(status)}")
            return False
        
        self.cooler_on = False
        return True
    
    def start_streaming(self):
        """Start continuous video streaming"""
        if not self.connected or self.streaming:
            return False
        
        status = StartExposure(self.camera_id, False)  # False for continuous mode
        if status != POAErrors.POA_OK:
            print(f"Error starting video capture: {GetErrorString(status)}")
            return False
        
        self.streaming = True
        return True
    
    def stop_streaming(self):
        """Stop video streaming"""
        if not self.connected or not self.streaming:
            return False
        
        status = StopExposure(self.camera_id)
        if status != POAErrors.POA_OK:
            print(f"Error stopping video capture: {GetErrorString(status)}")
            return False
        
        self.streaming = False
        return True
    
    def get_frame(self, timeout_ms=500):
        """Get a frame from the camera using direct image acquisition"""
        if not self.connected or not self.streaming:
            return None
        
        # Get frame size and format info
        status, width, height = GetImageSize(self.camera_id)
        if status != POAErrors.POA_OK:
            print(f"Error getting image size: {GetErrorString(status)}")
            return None
        
        status, img_format = GetImageFormat(self.camera_id)
        if status != POAErrors.POA_OK:
            print(f"Error getting image format: {GetErrorString(status)}")
            return None
        
        # Calculate buffer size and prepare buffer
        img_size = ImageCalcSize(height, width, img_format)
        buf_array = np.zeros(img_size, dtype=np.uint8)
        
        # Check if image is ready before retrieving it
        status, is_ready = ImageReady(self.camera_id)
        if status != POAErrors.POA_OK:
            print(f"Error checking if image is ready: {GetErrorString(status)}")
            return None
            
        if not is_ready:
            # If not ready within timeout, return None
            start_time = time.time()
            while not is_ready and (time.time() - start_time) * 1000 < timeout_ms:
                status, is_ready = ImageReady(self.camera_id)
                if status != POAErrors.POA_OK:
                    break
                time.sleep(0.001)  # Small sleep to prevent CPU hogging
                
            if not is_ready:
                print("Timeout waiting for image to be ready")
                return None
        
        # Get image data into buffer
        status = GetImageData(self.camera_id, buf_array, timeout_ms)
        if status != POAErrors.POA_OK:
            print(f"Error getting image data: {GetErrorString(status)}")
            return None
            
        # Convert buffer to image
        img = ImageDataConvert(buf_array, height, width, img_format)
        
        # Handle different image formats
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.reshape((img.shape[0], img.shape[1]))
            
        return img
