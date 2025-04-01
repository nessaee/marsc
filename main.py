#!/usr/bin/env python3

import cv2
import numpy as np
import time
import argparse
import sys
import os
import platform
import threading
import queue

# Ensure the pyPOACamera module is available
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import existing pyPOACamera module
try:
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
        """Connect to camera"""
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
    
    def set_binning(self, bin_value):
        """Set binning"""
        if not self.connected:
            return False
        
        was_streaming = self.streaming
        if was_streaming:
            self.stop_streaming()
        
        status = SetImageBin(self.camera_id, bin_value)
        if status != POAErrors.POA_OK:
            print(f"Error setting binning: {GetErrorString(status)}")
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
        """Get a frame from the camera"""
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
        
        # Calculate buffer size
        img_size = ImageCalcSize(height, width, img_format)
        
        # Get frame data
        status, img = GetImage(self.camera_id, timeout_ms)
        if status != POAErrors.POA_OK:
            if status == POAErrors.POA_ERROR_TIMEOUT:
                print("Timeout waiting for image")
            else:
                print(f"Error getting image: {GetErrorString(status)}")
            return None
        
        # Handle 3D array with single channel
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.reshape((img.shape[0], img.shape[1]))
            
        return img

def main():
    parser = argparse.ArgumentParser(description='Mars Camera Viewer')
    parser.add_argument('--sn', help='Camera serial number')
    parser.add_argument('--index', type=int, default=0, help='Camera index (0-based)')
    parser.add_argument('--exposure', type=float, default=10.0, help='Exposure time (ms)')
    parser.add_argument('--gain', type=float, default=0, help='Gain value')
    parser.add_argument('--offset', type=int, default=10, help='Black level offset')
    parser.add_argument('--bin', type=int, default=4, choices=[1, 2, 3, 4], help='Binning')
    parser.add_argument('--usb-limit', type=int, default=80, help='USB bandwidth limit (35-100%)')
    parser.add_argument('--cooler', type=int, default=-10, help='Cooler target temperature (C)')
    parser.add_argument('--save-dir', default='.', help='Directory for saving images')
    args = parser.parse_args()
    
    camera = None
    try:
        # List available cameras
        cameras = MarsCamera.list_cameras()
        if not cameras:
            print("No Player One cameras found")
            return
        
        print(f"Found {len(cameras)} camera(s):")
        for i, cam in enumerate(cameras):
            print(f"[{i}] {cam['name']} (S/N: {cam['sn']})")
        
        # Connect to camera
        camera = MarsCamera(serial_number=args.sn, index=args.index)
        if not camera.connect():
            print("Failed to connect to camera")
            return
        
        print(f"Connected to {camera.get_name()} (S/N: {camera.get_sn()})")
        
        # Configure camera
        camera.set_binning(args.bin)
        width, height = camera.get_size()
        print(f"Resolution: {width}x{height}")
        
        camera.set_roi(0, 0, width, height)
        camera.set_exposure(args.exposure)
        camera.set_gain(args.gain)
        camera.set_offset(args.offset)
        camera.set_usb_limit(args.usb_limit)
        
        # Configure cooler if available
        if camera.has_cooler():
            camera.set_target_temp(args.cooler)
            camera.cooler_on()
            print(f"Cooler enabled, target: {args.cooler}°C, current: {camera.get_temp():.1f}°C")
        
        # Setup display window
        cv2.namedWindow('Mars Camera', cv2.WINDOW_NORMAL)
        
        # Setup image processing parameters
        display_mode = 1  # 0=Auto, 1=Full range, 2=Native, 3=Bit shift
        contrast_mode = 0  # 0=Normal, 1=Enhanced, 2=High
        grid_mode = 0  # 0=None, 1=Rule of Thirds, 2=Golden Ratio, 3=Fine Grid
        rms_values = []
        max_rms = 20
        
        # Setup FPS calculation
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # Current settings for interactive control
        current_exposure = args.exposure
        current_gain = args.gain
        current_bin = args.bin
        
        # Start streaming
        if not camera.start_streaming():
            print("Failed to start streaming")
            return
        
        print("\nStreaming... Keyboard controls:")
        print("ESC - Exit program")
        print("s   - Save current image")
        print("h   - Show this help")
        print("+ / - - Increase/decrease exposure time")
        print("g / G - Decrease/increase gain")
        print("b / B - Decrease/increase binning")
        print("d   - Cycle through display modes")
        print("c   - Cycle through contrast modes")
        print("g   - Cycle through grid overlay modes")
        print("r   - Reset RMS plot scaling")
        
        # Setup frame queue for multithreading
        frame_queue = queue.Queue(maxsize=5)  # Limit queue size to prevent memory issues
        processing_fps = 0
        acquisition_fps = 0
        running = True
        
        # Frame acquisition thread function
        def frame_acquisition_thread():
            nonlocal acquisition_fps, running
            acq_frame_count = 0
            acq_start_time = time.time()
            
            while running:
                # Get frame
                frame = camera.get_frame(timeout_ms=500)  # Reduced timeout for faster response
                if frame is not None:
                    # If queue is full, remove oldest frame to make room
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    # Add new frame to queue
                    try:
                        frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass
                    
                    # Calculate acquisition FPS
                    acq_frame_count += 1
                    if acq_frame_count >= 10:
                        acq_end_time = time.time()
                        acquisition_fps = acq_frame_count / (acq_end_time - acq_start_time)
                        acq_frame_count = 0
                        acq_start_time = time.time()
        
        # Start frame acquisition thread
        acq_thread = threading.Thread(target=frame_acquisition_thread, daemon=True)
        acq_thread.start()
        
        # Main processing and display loop
        while running:
            # Get frame from queue
            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                # No frame available, check for key presses and continue
                k = cv2.waitKey(1)
                if k == 27:  # ESC key
                    running = False
                    break
                continue
            
            # Calculate processing FPS
            frame_count += 1
            if frame_count >= 10:
                end_time = time.time()
                processing_fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Keep original for processing and saving
            original = frame.copy()
            
            # Get bit depth and value range info
            bit_depth = camera.get_bit_depth()
            frame_min = np.min(frame)
            frame_max = np.max(frame)
            dynamic_range = frame_max - frame_min
            max_theoretical_val = float((1 << min(bit_depth, 24)) - 1)
            
            # Process image based on display mode
            if display_mode == 1:  # Full dynamic range
                if dynamic_range > 0:
                    gray = np.clip(((frame - frame_min) * 255.0 / dynamic_range), 0, 255).astype(np.uint8)
                else:
                    gray = np.zeros_like(frame, dtype=np.uint8)
                scaling_method = "Full dynamic range"
            elif display_mode == 2:  # Native bit depth
                if bit_depth <= 8:
                    gray = frame.astype(np.uint8)
                else:
                    scale_factor = 255.0 / max_theoretical_val
                    gray = np.clip(frame * scale_factor, 0, 255).astype(np.uint8)
                scaling_method = "Native bit depth"
            elif display_mode == 3:  # Bit shift
                if bit_depth <= 8:
                    gray = frame.astype(np.uint8)
                else:
                    shift_amount = bit_depth - 8
                    gray = (frame >> shift_amount).astype(np.uint8)
                scaling_method = f"Bit shift ({bit_depth}→8-bit)"
            else:  # Auto mode
                if dynamic_range > 50 and dynamic_range / max_theoretical_val > 0.1:
                    gray = np.clip(((frame - frame_min) * 255.0 / dynamic_range), 0, 255).astype(np.uint8)
                    scaling_method = "Auto: Dynamic range"
                else:
                    if bit_depth <= 8:
                        gray = frame.astype(np.uint8)
                    else:
                        scale_factor = 255.0 / max_theoretical_val
                        gray = np.clip(frame * scale_factor, 0, 255).astype(np.uint8)
                    scaling_method = "Auto: Native bit depth"
            
            # Apply contrast enhancement if enabled
            if contrast_mode == 1:  # Enhanced contrast
                p_low, p_high = np.percentile(gray, [5, 95])
                if p_high > p_low:
                    gray = np.clip(255.0 * (gray - p_low) / (p_high - p_low), 0, 255).astype(np.uint8)
                contrast_str = "Enhanced contrast"
            elif contrast_mode == 2:  # High contrast (histogram equalization)
                gray = cv2.equalizeHist(gray)
                contrast_str = "High contrast"
            else:  # Normal contrast
                contrast_str = "Normal contrast"

            # Apply Laplacian filter for focus measurement
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
            
            # Calculate RMS value (focus metric)
            rms = np.sqrt(np.mean(np.square(laplacian)))
            rms_values.append(rms)
            
            # Adjust max RMS value for better scaling if needed
            if rms > max_rms:
                max_rms = rms * 1.2
            
            # Keep only the last 100 values for display
            if len(rms_values) > 100:
                rms_values.pop(0)
            
            # Create display image
            display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Add grid overlay if enabled
            h, w = display_frame.shape[:2]
            if grid_mode == 1:  # Rule of Thirds
                # Vertical lines
                x1, x2 = int(w/3), int(2*w/3)
                cv2.line(display_frame, (x1, 0), (x1, h), (0, 140, 255), 1)
                cv2.line(display_frame, (x2, 0), (x2, h), (0, 140, 255), 1)
                # Horizontal lines
                y1, y2 = int(h/3), int(2*h/3)
                cv2.line(display_frame, (0, y1), (w, y1), (0, 140, 255), 1)
                cv2.line(display_frame, (0, y2), (w, y2), (0, 140, 255), 1)
            elif grid_mode == 2:  # Golden Ratio (Phi ≈ 1.618)
                # Vertical lines
                phi = 1.618
                x1 = int(w / (1 + phi))
                x2 = int(w - x1)
                cv2.line(display_frame, (x1, 0), (x1, h), (0, 140, 255), 1)
                cv2.line(display_frame, (x2, 0), (x2, h), (0, 140, 255), 1)
                # Horizontal lines
                y1 = int(h / (1 + phi))
                y2 = int(h - y1)
                cv2.line(display_frame, (0, y1), (w, y1), (0, 140, 255), 1)
                cv2.line(display_frame, (0, y2), (w, y2), (0, 140, 255), 1)
            elif grid_mode == 3:  # Fine Grid
                # Draw a 5x5 grid
                for i in range(1, 5):
                    # Vertical lines
                    x = int(i * w / 5)
                    cv2.line(display_frame, (x, 0), (x, h), (0, 140, 255), 1)
                    # Horizontal lines
                    y = int(i * h / 5)
                    cv2.line(display_frame, (0, y), (w, y), (0, 140, 255), 1)
            
            # Add info overlay
            percent_of_max = min(100, int(100.0 * dynamic_range / max_theoretical_val))
            
            cv2.putText(display_frame, f"Display FPS: {processing_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Camera FPS: {acquisition_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"RMS: {rms:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Exp: {current_exposure:.1f}ms", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Gain: {current_gain:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Bin: {current_bin}x{current_bin}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Get grid mode string
            grid_modes = ["No Grid", "Rule of Thirds", "Golden Ratio", "Fine Grid"]
            grid_str = grid_modes[grid_mode]
            
            cv2.putText(display_frame, f"{scaling_method}, {contrast_str}, {grid_str}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Range: {frame_min}-{frame_max} ({percent_of_max}%)", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if camera.has_cooler():
                current_temp = camera.get_temp()
                if current_temp is not None:
                    cv2.putText(display_frame, f"Temp: {current_temp:.1f}°C", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw RMS plot
            h, w = display_frame.shape[:2]
            plot_h, plot_w = 100, w
            plot_y = h - plot_h - 10
            
            if plot_y > 0:
                # Draw plot background
                cv2.rectangle(display_frame, (0, plot_y), (plot_w, plot_y + plot_h), (20, 20, 20), -1)
                
                # Draw RMS trend if we have values
                if len(rms_values) > 1:
                    for i in range(1, len(rms_values)):
                        y1 = plot_y + plot_h - int((rms_values[i-1] / max_rms) * (plot_h - 10))
                        y2 = plot_y + plot_h - int((rms_values[i] / max_rms) * (plot_h - 10))
                        y1 = max(plot_y, min(plot_y + plot_h, y1))
                        y2 = max(plot_y, min(plot_y + plot_h, y2))
                        x1 = int((i-1) * plot_w / max(1, len(rms_values)-1))
                        x2 = int(i * plot_w / max(1, len(rms_values)-1))
                        cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    
                    # Draw horizontal line at current RMS value
                    current_y = plot_y + plot_h - int((rms / max_rms) * (plot_h - 10))
                    current_y = max(plot_y, min(plot_y + plot_h, current_y))
                    cv2.line(display_frame, (0, current_y), (plot_w, current_y), (0, 128, 255), 1)
                
                # Add scale to RMS plot
                cv2.putText(display_frame, f"Focus Trend (Max: {max_rms:.1f})", 
                           (10, plot_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Mars Camera', display_frame)
            
            # Check for key press
            k = cv2.waitKey(1)
            if k == 27:  # ESC key
                running = False
                break
            elif k == ord('s'):  # Save image
                # Create save directory if it doesn't exist
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                base_filename = os.path.join(args.save_dir, f"mars_capture_{timestamp}")
                
                # Save original bit depth image if available
                if bit_depth > 8 and 'original' in locals():
                    if bit_depth <= 16:
                        cv2.imwrite(f"{base_filename}.png", original)
                        print(f"Full bit depth image saved as {base_filename}.png")
                
                # Save 8-bit image
                cv2.imwrite(f"{base_filename}_8bit.png", gray)
                print(f"8-bit image saved as {base_filename}_8bit.png")
                
                # Save stretched version for better visibility
                stretch_img = cv2.normalize(gray, None, alpha=0, beta=255, 
                                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(f"{base_filename}_stretched.png", stretch_img)
                print(f"Enhanced image saved as {base_filename}_stretched.png")
                
                # Save display view
                cv2.imwrite(f"{base_filename}_display.jpg", display_frame)
                print(f"Display view saved as {base_filename}_display.jpg")
                
            elif k == ord('h'):  # Help
                print("\nKeyboard controls:")
                print("ESC - Exit program")
                print("s   - Save current image")
                print("h   - Show this help")
                print("+ / - - Increase/decrease exposure time")
                print("g / G - Decrease/increase gain")
                print("b / B - Decrease/increase binning")
                print("d   - Cycle through display modes (Auto/Dynamic Range/Native/Bit-shift)")
                print("c   - Cycle through contrast modes (Normal/Enhanced/High)")
                print("g   - Cycle through grid modes (None/Rule of Thirds/Golden Ratio/Fine Grid)")
                print("r   - Reset RMS plot scaling")
            elif k == ord('d'):  # Change display mode
                display_mode = (display_mode + 1) % 4
                mode_names = ["Auto", "Full dynamic range", "Native bit depth", "Bit shift"]
                print(f"Display mode changed to: {mode_names[display_mode]}")
            elif k == ord('c'):  # Change contrast mode
                contrast_mode = (contrast_mode + 1) % 3
                contrast_modes = ["Normal", "Enhanced", "High contrast"]
                print(f"Contrast mode changed to: {contrast_modes[contrast_mode]}")
            elif k == ord('g'):  # Change grid mode
                grid_mode = (grid_mode + 1) % 4
                grid_modes = ["None", "Rule of Thirds", "Golden Ratio", "Fine Grid"]
                print(f"Grid mode changed to: {grid_modes[grid_mode]}")
            elif k == ord('r'):  # Reset RMS plot scaling
                max_rms = 20
                rms_values = []
                print("RMS plot scaling reset")
            elif k == ord('+') or k == ord('='):  # Increase exposure
                current_exposure = min(5000, current_exposure * 1.2)
                camera.set_exposure(current_exposure)
                print(f"Exposure increased to {current_exposure:.1f}ms")
            elif k == ord('-'):  # Decrease exposure
                current_exposure = max(0.1, current_exposure / 1.2)
                camera.set_exposure(current_exposure)
                print(f"Exposure decreased to {current_exposure:.1f}ms")
            elif k == ord('g'):  # Decrease gain
                current_gain = max(0, current_gain - 1)
                camera.set_gain(current_gain)
                print(f"Gain decreased to {current_gain:.1f}")
            elif k == ord('G'):  # Increase gain
                current_gain = min(100, current_gain + 1)
                camera.set_gain(current_gain)
                print(f"Gain increased to {current_gain:.1f}")
            elif k == ord('b'):  # Decrease binning
                new_bin = max(1, current_bin - 1)
                if new_bin != current_bin:
                    current_bin = new_bin
                    camera.stop_streaming()
                    camera.set_binning(current_bin)
                    width, height = camera.get_size()
                    print(f"Binning decreased to {current_bin}x{current_bin}")
                    print(f"New resolution: {width}x{height}")
                    camera.start_streaming()
            elif k == ord('B'):  # Increase binning
                new_bin = min(4, current_bin + 1)
                if new_bin != current_bin:
                    current_bin = new_bin
                    camera.stop_streaming()
                    camera.set_binning(current_bin)
                    width, height = camera.get_size()
                    print(f"Binning increased to {current_bin}x{current_bin}")
                    print(f"New resolution: {width}x{height}")
                    camera.start_streaming()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Signal threads to stop
        running = False
        
        # Wait for acquisition thread to finish
        if 'acq_thread' in locals() and acq_thread.is_alive():
            acq_thread.join(timeout=1.0)
            
        # Clean up
        if camera is not None:
            if camera.streaming:
                camera.stop_streaming()
                print("Streaming stopped")
            if camera.has_cooler() and camera.cooler_on:
                camera.cooler_off()
                print("Cooler turned off")
            if camera.connected:
                camera.disconnect()
                print("Camera disconnected")
        
        cv2.destroyAllWindows()
        print("Application terminated")

if __name__ == "__main__":
    main()
