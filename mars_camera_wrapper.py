#!/usr/bin/env python3

import os
import sys
import time
import platform
import numpy as np
from ctypes import *

class POABayerPattern:
    '''Bayer Pattern Definition'''
    POA_BAYER_MONO = -1         # Monochrome, the mono camera with this
    POA_BAYER_RG = 0            # RGGB
    POA_BAYER_BG = 1            # BGGR
    POA_BAYER_GR = 2            # GRBG
    POA_BAYER_GB = 3            # GBRG

class POAImgFormat:
    '''Image Data Format Definition'''
    POA_END = -1                # ending in imgFormats[] of POACameraProperties, please ignore this
    POA_RAW8 = 0                # 8bit raw data, 1 pixel 1 byte, value range[0, 255]
    POA_RAW16 = 1               # 16bit raw data, 1 pixel 2 bytes, value range[0, 65535]
    POA_RGB24 = 2               # RGB888 color data, 1 pixel 3 bytes, value range[0, 255] (only color camera)
    POA_MONO8 = 3               # 8bit monochrome data, convert the Bayer Filter Array to monochrome data. 1 pixel 1 byte, value range[0, 255] (only color camera)

class POAErrors:
    '''Return Error Code Definition'''
    POA_OK = 0                              # operation successful
    POA_ERROR_INVALID_INDEX = 1             # invalid index, means the index is < 0 or >= the count( camera or config)
    POA_ERROR_INVALID_ID = 2                # invalid camera ID
    POA_ERROR_INVALID_CONFIG = 3            # invalid POAConfig
    POA_ERROR_INVALID_ARGU = 4              # invalid argument(parameter)
    POA_ERROR_NOT_OPENED = 5                # camera not opened
    POA_ERROR_DEVICE_NOT_FOUND = 6          # camera not found, may be removed
    POA_ERROR_OUT_OF_LIMIT = 7              # the value out of limit
    POA_ERROR_EXPOSURE_FAILED = 8           # camera exposure failed
    POA_ERROR_TIMEOUT = 9                   # timeout
    POA_ERROR_SIZE_LESS = 10                # the data buffer size is not enough
    POA_ERROR_EXPOSING = 11                 # camera is exposing. some operation, must stop exposure first
    POA_ERROR_POINTER = 12                  # invalid pointer, when get some value, do not pass the NULL pointer to the function
    POA_ERROR_CONF_CANNOT_WRITE = 13        # the POAConfig is not writable
    POA_ERROR_CONF_CANNOT_READ = 14         # the POAConfig is not readable
    POA_ERROR_ACCESS_DENIED = 15            # access denied
    POA_ERROR_OPERATION_FAILED = 16         # operation failed, maybe the camera is disconnected suddenly
    POA_ERROR_MEMORY_FAILED = 17            # memory allocation failed

class POACameraState:
    '''Camera State Definition'''
    STATE_CLOSED = 0                # camera was closed
    STATE_OPENED = 1                # camera was opened, but not exposing(idle)
    STATE_EXPOSING = 2              # camera is exposing

class POAConfig:
    '''Camera Config Definition'''
    POA_EXPOSURE = 0                    # exposure time(microsecond (us)), range:[10 - 2000000000], read-write, support auto
    POA_GAIN = 1                        # gain, read-write, support auto
    POA_HARDWARE_BIN = 2                # hardware bin, read-write, On/Off type(bool)
    POA_WB_R = 4                        # red pixels coefficient of white balance, read-write
    POA_WB_G = 5                        # green pixels coefficient of white balance, read-write
    POA_WB_B = 6                        # blue pixels coefficient of white balance, read-write
    POA_OFFSET = 7                      # camera offset, read-write
    POA_AUTOEXPO_MAX_GAIN = 8           # maximum gain when auto-adjust, read-write
    POA_AUTOEXPO_MAX_EXPOSURE = 9       # maximum exposure when auto-adjust(uint: ms), read-write
    POA_AUTOEXPO_BRIGHTNESS = 10        # target brightness when auto-adjust, read-write
    POA_COOLER_POWER = 16               # cooler power percentage[0-100%](only cool camera), read-only
    POA_TARGET_TEMP = 17                # camera target temperature(uint: C), read-write
    POA_COOLER = 18                     # turn cooler(and fan) on or off, read-write, On/Off type(bool)
    POA_HEATER = 19                     # (deprecated)get state of lens heater(on or off), read-only
    POA_HEATER_POWER = 20               # lens heater power percentage[0-100%], read-write
    POA_FAN_POWER = 21                  # radiator fan power percentage[0-100%], read-write
    POA_FRAME_LIMIT = 26                # Frame rate limit, the range:[0, 2000], 0 means no limit, read-write
    POA_HQI = 27                        # High Quality Image, for those cameras without DDR(guide camera), reduce frame rate to improve image quality, read-write, On/Off type(bool)
    POA_USB_BANDWIDTH_LIMIT = 28        # USB bandwidth limit[35-100]%, read-write
    POA_PIXEL_BIN_SUM = 29              # take the sum of pixels after binning, True(1) is sum and False(0) is average, default is False(0), read-write, On/Off type(bool)
    POA_MONO_BIN = 30                   # only for color camera, when set to True, pixel binning will use neighbour pixels and image after binning will lose the bayer pattern, read-write, On/Off type(bool)

class PlayerOneCamera:
    def __init__(self, lib_path=None):
        """Initialize the PlayerOne camera interface with the specified library path"""
        self.dll = None
        self.lib_path = lib_path
        self.load_library()
        
    def load_library(self):
        """Load the PlayerOne camera library"""
        if self.lib_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            sdk_dir = os.path.join(script_dir, "playerone_sdk/PlayerOne_Camera_SDK_Linux_V3.8.1")
            
            # Determine architecture
            arch = platform.machine()
            if arch == "x86_64":
                lib_arch_dir = "x64"
            elif arch == "i386" or arch == "i686":
                lib_arch_dir = "x86"
            elif arch.startswith("arm") and "64" in arch:
                lib_arch_dir = "arm64"
            elif arch.startswith("arm"):
                lib_arch_dir = "arm32"
            else:
                raise Exception(f"Unsupported architecture: {arch}")
            
            self.lib_path = os.path.join(sdk_dir, "lib", lib_arch_dir, "libPlayerOneCamera.so")
        
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"Library not found at {self.lib_path}")
        
        # Add library directory to LD_LIBRARY_PATH
        lib_dir = os.path.dirname(self.lib_path)
        os.environ['LD_LIBRARY_PATH'] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        
        try:
            self.dll = cdll.LoadLibrary(self.lib_path)
            print(f"Successfully loaded library: {self.lib_path}")
        except Exception as e:
            raise Exception(f"Failed to load library: {e}")
    
    def get_camera_count(self):
        """Get the number of connected cameras"""
        self.dll.POAGetCameraCount.restype = c_int
        return self.dll.POAGetCameraCount()
    
    def get_camera_properties(self, index):
        """Get camera properties by index"""
        class POACameraProperties(Structure):
            _fields_ = [("cameraModelName", c_char * 256),
                       ("userCustomID", c_char * 16),
                       ("cameraID", c_int),
                       ("maxWidth", c_int),
                       ("maxHeight", c_int),
                       ("bitDepth", c_int),
                       ("isColorCamera", c_int),
                       ("isHasST4Port", c_int),
                       ("isHasCooler", c_int),
                       ("isUSB3Speed", c_int),
                       ("bayerPattern_", c_int),
                       ("pixelSize", c_double),
                       ("SN", c_char * 64),
                       ("sensorModelName", c_char * 32),
                       ("localPath", c_char * 256),
                       ("bins_", c_int * 8),
                       ("imgFormats_", c_int * 8),
                       ("isSupportHardBin", c_int),
                       ("pID", c_int),
                       ("reserved", c_char * 248)]
        
        props = POACameraProperties()
        func = self.dll.POAGetCameraProperties
        func.argtypes = [c_int, POINTER(POACameraProperties)]
        func.restype = c_int
        status = func(index, byref(props))
        
        if status != 0:  # POA_OK
            raise Exception(f"Error getting camera properties: {status}")
        
        # Convert to a dictionary for easier access
        result = {
            'name': props.cameraModelName.decode('utf-8').strip('\0'),
            'id': props.cameraID,
            'sn': props.SN.decode('utf-8').strip('\0'),
            'max_width': props.maxWidth,
            'max_height': props.maxHeight,
            'bit_depth': props.bitDepth,
            'is_color': bool(props.isColorCamera),
            'has_st4': bool(props.isHasST4Port),
            'has_cooler': bool(props.isHasCooler),
            'is_usb3': bool(props.isUSB3Speed),
            'bayer_pattern': props.bayerPattern_,
            'pixel_size': props.pixelSize,
            'sensor': props.sensorModelName.decode('utf-8').strip('\0'),
            'supports_hard_bin': bool(props.isSupportHardBin),
        }
        
        # Process bin values
        bins = []
        for i in range(8):
            if props.bins_[i] == 0:
                break
            bins.append(props.bins_[i])
        result['bins'] = bins
        
        # Process image formats
        formats = []
        for i in range(8):
            fmt = props.imgFormats_[i]
            if fmt == -1:  # POA_END
                break
            formats.append(fmt)
        result['formats'] = formats
        
        return result

class POACamera:
    def __init__(self, serial_number=None):
        """Initialize a camera instance"""
        self.sdk = PlayerOneCamera()
        self.serial_number = serial_number
        self.camera_id = None
        self.info = None
        self.connected = False
        self.streaming = False
        self.cooler_on_status = False
    
    @staticmethod
    def get_cameras():
        """Get list of all available cameras"""
        sdk = PlayerOneCamera()
        count = sdk.get_camera_count()
        cameras = []
        
        for i in range(count):
            try:
                camera_info = sdk.get_camera_properties(i)
                cameras.append(camera_info)
            except Exception as e:
                print(f"Error getting properties for camera {i}: {e}")
        
        return cameras
    
    def open(self):
        """Open the camera by serial number"""
        cameras = self.get_cameras()
        
        if not cameras:
            print("No cameras found")
            return False
        
        # Find camera by serial number if provided
        if self.serial_number:
            found = False
            for cam in cameras:
                if cam['sn'] == self.serial_number:
                    self.info = cam
                    self.camera_id = cam['id']
                    found = True
                    break
            
            if not found:
                print(f"Camera with serial number {self.serial_number} not found")
                return False
        else:
            # Use the first camera
            self.info = cameras[0]
            self.camera_id = cameras[0]['id']
            self.serial_number = cameras[0]['sn']
        
        # Open the camera
        status = self.sdk.dll.POAOpenCamera(self.camera_id)
        if status != 0:  # POA_OK
            print(f"Error opening camera: {status}")
            return False
        
        # Initialize the camera
        status = self.sdk.dll.POAInitCamera(self.camera_id)
        if status != 0:  # POA_OK
            print(f"Error initializing camera: {status}")
            self.sdk.dll.POACloseCamera(self.camera_id)
            return False
        
        self.connected = True
        return True
    
    def close(self):
        """Close the camera"""
        if not self.connected:
            return
        
        if self.streaming:
            self.stop_streaming()
        
        if self.cooler_on_status:
            self.cooler_off()
        
        status = self.sdk.dll.POACloseCamera(self.camera_id)
        self.connected = False
        return status == 0  # POA_OK
    
    def get_name(self):
        """Get camera name"""
        return self.info['name'] if self.info else None
    
    def get_sn(self):
        """Get camera serial number"""
        return self.serial_number
    
    def get_size(self):
        """Get current image size"""
        width = c_int(0)
        height = c_int(0)
        status = self.sdk.dll.POAGetImageSize(self.camera_id, byref(width), byref(height))
        if status != 0:
            raise Exception(f"Error getting image size: {status}")
        return width.value, height.value
    
    def set_roi(self, x, y, width, height):
        """Set region of interest"""
        if self.streaming:
            self.stop_streaming()
            restart_stream = True
        else:
            restart_stream = False
        
        status = self.sdk.dll.POASetImageStartPos(self.camera_id, x, y)
        if status != 0:
            raise Exception(f"Error setting image start position: {status}")
        
        status = self.sdk.dll.POASetImageSize(self.camera_id, width, height)
        if status != 0:
            raise Exception(f"Error setting image size: {status}")
        
        if restart_stream:
            self.start_streaming()
    
    def get_binning(self):
        """Get current binning"""
        bin_val = c_int(0)
        status = self.sdk.dll.POAGetImageBin(self.camera_id, byref(bin_val))
        if status != 0:
            raise Exception(f"Error getting image binning: {status}")
        return bin_val.value
    
    def set_binning(self, bin_value):
        """Set binning"""
        if self.streaming:
            self.stop_streaming()
            restart_stream = True
        else:
            restart_stream = False
        
        status = self.sdk.dll.POASetImageBin(self.camera_id, bin_value)
        if status != 0:
            raise Exception(f"Error setting image binning: {status}")
        
        if restart_stream:
            self.start_streaming()
    
    def get_bit_depth(self):
        """Get bit depth"""
        return self.info['bit_depth'] if self.info else 0
    
    def has_cooler(self):
        """Check if camera has cooler"""
        return self.info['has_cooler'] if self.info else False
    
    def is_connected(self):
        """Check if camera is connected"""
        return self.connected
    
    def is_streaming(self):
        """Check if camera is streaming"""
        return self.streaming
    
    def is_cooler_on(self):
        """Check if cooler is on"""
        return self.cooler_on_status
    
    def set_exposure(self, exposure_ms):
        """Set exposure time in milliseconds"""
        exposure_us = int(exposure_ms * 1000)  # Convert ms to us
        status = self.sdk.dll.POASetConfig(self.camera_id, POAConfig.POA_EXPOSURE, exposure_us, 0)
        if status != 0:
            raise Exception(f"Error setting exposure: {status}")
    
    def get_exposure(self):
        """Get exposure time in milliseconds"""
        exposure = c_long(0)
        is_auto = c_int(0)
        status = self.sdk.dll.POAGetConfig(self.camera_id, POAConfig.POA_EXPOSURE, byref(exposure), byref(is_auto))
        if status != 0:
            raise Exception(f"Error getting exposure: {status}")
        return exposure.value / 1000  # Convert us to ms
    
    def set_gain(self, gain):
        """Set gain"""
        status = self.sdk.dll.POASetConfig(self.camera_id, POAConfig.POA_GAIN, int(gain), 0)
        if status != 0:
            raise Exception(f"Error setting gain: {status}")
    
    def get_gain(self):
        """Get gain"""
        gain = c_long(0)
        is_auto = c_int(0)
        status = self.sdk.dll.POAGetConfig(self.camera_id, POAConfig.POA_GAIN, byref(gain), byref(is_auto))
        if status != 0:
            raise Exception(f"Error getting gain: {status}")
        return gain.value
    
    def set_offset(self, offset):
        """Set offset"""
        status = self.sdk.dll.POASetConfig(self.camera_id, POAConfig.POA_OFFSET, offset, 0)
        if status != 0:
            raise Exception(f"Error setting offset: {status}")
    
    def get_offset(self):
        """Get offset"""
        offset = c_long(0)
        is_auto = c_int(0)
        status = self.sdk.dll.POAGetConfig(self.camera_id, POAConfig.POA_OFFSET, byref(offset), byref(is_auto))
        if status != 0:
            raise Exception(f"Error getting offset: {status}")
        return offset.value
    
    def set_target_temp(self, temp):
        """Set target temperature"""
        if not self.has_cooler():
            print("Camera does not have a cooler")
            return
        
        status = self.sdk.dll.POASetConfig(self.camera_id, POAConfig.POA_TARGET_TEMP, temp, 0)
        if status != 0:
            raise Exception(f"Error setting target temperature: {status}")
    
    def get_target_temp(self):
        """Get target temperature"""
        if not self.has_cooler():
            return None
        
        temp = c_long(0)
        is_auto = c_int(0)
        status = self.sdk.dll.POAGetConfig(self.camera_id, POAConfig.POA_TARGET_TEMP, byref(temp), byref(is_auto))
        if status != 0:
            raise Exception(f"Error getting target temperature: {status}")
        return temp.value
    
    def get_temp(self):
        """Get current temperature"""
        if not self.has_cooler():
            return None
        
        temp = c_double(0)
        is_auto = c_int(0)
        status = self.sdk.dll.POAGetConfig(self.camera_id, 3, byref(temp), byref(is_auto))  # POA_TEMPERATURE = 3
        if status != 0:
            raise Exception(f"Error getting temperature: {status}")
        return temp.value
    
    def cooler_on(self):
        """Turn cooler on"""
        if not self.has_cooler():
            print("Camera does not have a cooler")
            return
        
        status = self.sdk.dll.POASetConfig(self.camera_id, POAConfig.POA_COOLER, 1, 0)
        if status != 0:
            raise Exception(f"Error turning cooler on: {status}")
        self.cooler_on_status = True
    
    def cooler_off(self):
        """Turn cooler off"""
        if not self.has_cooler():
            return
        
        status = self.sdk.dll.POASetConfig(self.camera_id, POAConfig.POA_COOLER, 0, 0)
        if status != 0:
            raise Exception(f"Error turning cooler off: {status}")
        self.cooler_on_status = False
    
    def start_video(self):
        """Start continuous video capture"""
        if self.streaming:
            return
        
        status = self.sdk.dll.POAStartExposure(self.camera_id, 0)  # 0 = continuous mode
        if status != 0:
            raise Exception(f"Error starting video capture: {status}")
        self.streaming = True
    
    def start_streaming(self):
        """Alias for start_video"""
        self.start_video()
    
    def stop_video(self):
        """Stop video capture"""
        if not self.streaming:
            return
        
        status = self.sdk.dll.POAStopExposure(self.camera_id)
        if status != 0:
            raise Exception(f"Error stopping video capture: {status}")
        self.streaming = False
    
    def stop_streaming(self):
        """Alias for stop_video"""
        self.stop_video()
    
    def get_video_data(self, timeout_ms):
        """Get video frame data"""
        if not self.streaming:
            print("Camera is not streaming")
            return None
        
        # Get image size and format
        width, height = self.get_size()
        img_format = c_int(0)
        status = self.sdk.dll.POAGetImageFormat(self.camera_id, byref(img_format))
        if status != 0:
            raise Exception(f"Error getting image format: {status}")
        
        # Calculate buffer size
        if img_format.value == POAImgFormat.POA_RAW8 or img_format.value == POAImgFormat.POA_MONO8:
            buf_size = width * height
            dtype = np.uint8
        elif img_format.value == POAImgFormat.POA_RAW16:
            buf_size = width * height * 2
            dtype = np.uint16
        elif img_format.value == POAImgFormat.POA_RGB24:
            buf_size = width * height * 3
            dtype = np.uint8
        else:
            raise Exception(f"Unsupported image format: {img_format.value}")
        
        # Create buffer for image data
        if img_format.value == POAImgFormat.POA_RAW16:
            # For 16-bit data, allocate as uint8 first, then reshape
            buffer = np.zeros(buf_size, dtype=np.uint8)
            data_ptr = buffer.ctypes.data_as(POINTER(c_uint8))
            
            # Get image data
            status = self.sdk.dll.POAGetImageData(self.camera_id, data_ptr, buf_size, timeout_ms)
            if status != 0:
                if status == POAErrors.POA_ERROR_TIMEOUT:
                    print("Timeout waiting for image")
                    return None
                raise Exception(f"Error getting image data: {status}")
            
            # Convert to 16-bit array
            result = np.frombuffer(buffer.tobytes(), dtype=np.uint16, count=width*height)
            return result.reshape((height, width))
        else:
            # For 8-bit data
            buffer = np.zeros(buf_size, dtype=dtype)
            data_ptr = buffer.ctypes.data_as(POINTER(c_uint8))
            
            # Get image data
            status = self.sdk.dll.POAGetImageData(self.camera_id, data_ptr, buf_size, timeout_ms)
            if status != 0:
                if status == POAErrors.POA_ERROR_TIMEOUT:
                    print("Timeout waiting for image")
                    return None
                raise Exception(f"Error getting image data: {status}")
            
            # Reshape based on format
            if img_format.value == POAImgFormat.POA_RGB24:
                return buffer.reshape((height, width, 3))
            else:
                return buffer.reshape((height, width))

# For testing
if __name__ == "__main__":
    # List cameras
    cameras = POACamera.get_cameras()
    print(f"Found {len(cameras)} cameras:")
    for i, cam in enumerate(cameras):
        print(f"  {i}: {cam['name']} (S/N: {cam['sn']})")
    
    if len(cameras) > 0:
        # Connect to first camera
        camera = POACamera()
        if camera.open():
            print(f"Connected to: {camera.get_name()} (S/N: {camera.get_sn()})")
            print(f"Size: {camera.get_size()}")
            print(f"Bit depth: {camera.get_bit_depth()}")
            
            if camera.has_cooler():
                print(f"Temperature: {camera.get_temp():.1f}°C")
            
            camera.close()
            print("Camera closed")