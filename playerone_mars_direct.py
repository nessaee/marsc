#!/usr/bin/env python3

import cv2
import numpy as np
import time
import argparse
import sys
import os
import platform
from ctypes import cdll

# -----------------------------------------------------------------------------
# Initialize SDK environment
# -----------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sdk_dir = os.path.join(script_dir, "playerone_sdk/PlayerOne_Camera_SDK_Linux_V3.8.1")

# Make sure the Python module directory is in the path
python_dir = os.path.join(sdk_dir, "python")
sys.path.insert(0, python_dir)

# Determine platform architecture for the correct library path
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
    print(f"Unsupported architecture: {arch}")
    print("This script supports x64, x86, arm64, and arm32 architectures.")
    sys.exit(1)

# Path to the shared library
lib_dir = os.path.join(sdk_dir, "lib", lib_arch_dir)
lib_path = os.path.join(lib_dir, "libPlayerOneCamera.so")

# Check if library exists
if not os.path.exists(lib_path):
    print(f"ERROR: Player One camera library not found at {lib_path}")
    print("Please make sure the SDK is properly extracted")
    sys.exit(1)

# Add library directory to LD_LIBRARY_PATH
os.environ['LD_LIBRARY_PATH'] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
print(f"Using library: {lib_path}")

# -----------------------------------------------------------------------------
# Patch the pyPOACamera module to use our library path
# -----------------------------------------------------------------------------
# Copy and modify the module rather than directly modifying the original
pypoa_path = os.path.join(python_dir, "pyPOACamera.py")
local_pypoa_path = os.path.join(script_dir, "pyPOACamera_patched.py")

with open(pypoa_path, 'r') as f:
    pypoa_content = f.read()

# Replace the DLL loading line
pypoa_content = pypoa_content.replace(
    'dll = cdll.LoadLibrary("./PlayerOneCamera.dll")', 
    f'dll = cdll.LoadLibrary("{lib_path}")'
)
# Comment out the other loading lines
pypoa_content = pypoa_content.replace(
    '#dll = cdll.LoadLibrary("./libPlayerOneCamera.so")', 
    '#dll = cdll.LoadLibrary("./libPlayerOneCamera.so") # Original Linux line'
)

with open(local_pypoa_path, 'w') as f:
    f.write(pypoa_content)

# Import the patched module
sys.path.insert(0, script_dir)  # Make sure our directory is first in path
try:
    from pyPOACamera_patched import POACamera, POAConfig, POAImgFormat, POAErrors, POACameraState
    print("Successfully imported patched pyPOACamera module")
except ImportError as e:
    print(f"Error importing patched pyPOACamera module: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading Player One camera library: {e}")
    print(f"Library path: {lib_path}")
    print("\nPossible solutions:")
    print("1. Install required dependencies: sudo apt-get install libusb-1.0-0")
    print(f"2. Set up udev rules: sudo cp {os.path.join(sdk_dir, 'udev', '99-player_one_astronomy.rules')} /etc/udev/rules.d/")
    print("3. Run: sudo udevadm control --reload-rules && sudo udevadm trigger")
    print("4. Disconnect and reconnect your camera")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Helper class for camera operations
# -----------------------------------------------------------------------------
class MarsCamera:
    def __init__(self, camera_sn=None, camera_idx=0):
        self.camera = None
        self.camera_sn = camera_sn
        self.camera_idx = camera_idx
        self.connected = False
        self.width = 0
        self.height = 0
        self.bit_depth = 0
        self.has_cooler = False
    
    def list_cameras(self):
        """List all available cameras"""
        return POACamera.get_cameras()
    
    def connect(self):
        """Connect to the camera"""
        try:
            cameras = self.list_cameras()
            if not cameras:
                print("No Player One cameras found. Please check connection.")
                return False
                
            print(f"Found {len(cameras)} camera(s):")
            for i, cam in enumerate(cameras):
                print(f"[{i}] {cam.get('name')} (S/N: {cam.get('sn')})")
            
            # Use the specified index or serial number
            if self.camera_sn:
                for cam in cameras:
                    if cam.get('sn') == self.camera_sn:
                        self.camera = POACamera(self.camera_sn)
                        break
                if not self.camera:
                    print(f"Camera with S/N {self.camera_sn} not found, using first available.")
                    self.camera = POACamera(cameras[0].get('sn'))
            else:
                idx = min(self.camera_idx, len(cameras)-1)
                self.camera = POACamera(cameras[idx].get('sn'))
            
            # Connect to the camera
            result = self.camera.connect()
            if not result:
                print("Failed to connect to camera")
                return False
            
            self.connected = True
            self.has_cooler = self.camera.has_cooler()
            
            # Get camera info
            name = self.camera.get_name()
            sn = self.camera.get_sn()
            self.width, self.height = self.camera.get_size()
            self.bit_depth = self.camera.get_bit_depth()
            
            print(f"Connected to: {name} (S/N: {sn})")
            print(f"Resolution: {self.width}x{self.height}, {self.bit_depth}-bit")
            if self.has_cooler:
                print(f"Temperature: {self.camera.get_temp():.1f}°C")
            
            return True
            
        except Exception as e:
            print(f"Error connecting to camera: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def disconnect(self):
        """Disconnect from the camera"""
        if self.connected and self.camera:
            if self.camera.is_streaming():
                self.camera.stop_video()
            if self.has_cooler and self.camera.is_cooler_on():
                self.camera.cooler_off()
            self.camera.disconnect()
            self.connected = False
            print("Camera disconnected")
    
    def start_streaming(self):
        """Start video streaming"""
        if not self.connected:
            print("Camera not connected")
            return False
        
        try:
            self.camera.start_video()
            print("Streaming started")
            return True
        except Exception as e:
            print(f"Error starting streaming: {e}")
            return False
    
    def stop_streaming(self):
        """Stop video streaming"""
        if self.connected and self.camera.is_streaming():
            self.camera.stop_video()
            print("Streaming stopped")
    
    def get_frame(self, timeout_ms=5000):
        """Get a frame from the camera"""
        if not self.connected:
            return None
        
        try:
            frame = self.camera.get_video_data(timeout_ms)
            return frame
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None
    
    def set_roi(self, x, y, width, height):
        """Set region of interest"""
        if not self.connected:
            return False
        
        try:
            self.camera.set_roi(x, y, width, height)
            self.width, self.height = self.camera.get_size()
            print(f"ROI set to {self.width}x{self.height} starting at ({x},{y})")
            return True
        except Exception as e:
            print(f"Error setting ROI: {e}")
            return False
    
    def set_binning(self, bin_value):
        """Set binning"""
        if not self.connected:
            return False
        
        try:
            self.camera.set_binning(bin_value)
            self.width, self.height = self.camera.get_size()
            print(f"Binning set to {bin_value}x{bin_value}")
            print(f"New resolution: {self.width}x{self.height}")
            return True
        except Exception as e:
            print(f"Error setting binning: {e}")
            return False
    
    def set_exposure(self, exposure_ms):
        """Set exposure time in milliseconds"""
        if not self.connected:
            return False
        
        try:
            self.camera.set_exposure(exposure_ms)
            print(f"Exposure set to {exposure_ms}ms")
            return True
        except Exception as e:
            print(f"Error setting exposure: {e}")
            return False
    
    def set_gain(self, gain):
        """Set gain"""
        if not self.connected:
            return False
        
        try:
            self.camera.set_gain(gain)
            print(f"Gain set to {gain}")
            return True
        except Exception as e:
            print(f"Error setting gain: {e}")
            return False
    
    def set_offset(self, offset):
        """Set offset"""
        if not self.connected:
            return False
        
        try:
            self.camera.set_offset(offset)
            print(f"Offset set to {offset}")
            return True
        except Exception as e:
            print(f"Error setting offset: {e}")
            return False
    
    def enable_cooler(self, target_temp):
        """Enable cooler with target temperature"""
        if not self.connected or not self.has_cooler:
            return False
        
        try:
            self.camera.set_target_temp(target_temp)
            self.camera.cooler_on()
            print(f"Cooler enabled with target temperature {target_temp}°C")
            return True
        except Exception as e:
            print(f"Error enabling cooler: {e}")
            return False
    
    def disable_cooler(self):
        """Disable cooler"""
        if not self.connected or not self.has_cooler:
            return False
        
        try:
            self.camera.cooler_off()
            print("Cooler disabled")
            return True
        except Exception as e:
            print(f"Error disabling cooler: {e}")
            return False
    
    def get_temperature(self):
        """Get current temperature"""
        if not self.connected or not self.has_cooler:
            return None
        
        try:
            return self.camera.get_temp()
        except Exception as e:
            print(f"Error getting temperature: {e}")
            return None

# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Stream from Player One Mars camera with Laplacian filtering and RMS calculation')
    parser.add_argument('--sn', help='Camera serial number', default=None)
    parser.add_argument('--index', type=int, help='Camera index (0-based)', default=0)
    parser.add_argument('--exposure', type=int, help='Exposure time in milliseconds', default=10)
    parser.add_argument('--gain', type=float, help='Gain value (0-100)', default=20.0)
    parser.add_argument('--offset', type=int, help='Black level offset', default=10)
    parser.add_argument('--bin', type=int, help='Binning (1, 2, 3, 4)', default=1, choices=[1, 2, 3, 4])
    parser.add_argument('--cooler', type=int, help='Cooler target temperature (Celsius)', default=-10)
    parser.add_argument('--save-dir', help='Directory to save images', default='.')
    args = parser.parse_args()

    # Initialize the camera
    camera = MarsCamera(camera_sn=args.sn, camera_idx=args.index)
    
    try:
        # Connect to the camera
        if not camera.connect():
            sys.exit(1)
        
        # Configure camera settings
        camera.set_binning(args.bin)
        camera.set_roi(0, 0, camera.width, camera.height)  # Full frame
        camera.set_exposure(args.exposure)
        camera.set_gain(args.gain)
        camera.set_offset(args.offset)
        
        # Enable cooler if available
        if camera.has_cooler:
            camera.enable_cooler(args.cooler)
        
        # Start streaming
        if not camera.start_streaming():
            sys.exit(1)
        
        # Create display windows
        cv2.namedWindow('Mars Camera', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Laplacian Filter', cv2.WINDOW_NORMAL)
        cv2.namedWindow('RMS Trend', cv2.WINDOW_NORMAL)
        
        # Variables for FPS calculation
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # RMS values history for display
        rms_values = []
        max_rms = 20  # Initial scale for RMS plot
        
        # Current settings for interactive control
        current_exposure = args.exposure
        current_gain = args.gain
        
        print("\nStreaming... Keyboard controls:")
        print("ESC - Exit program")
        print("s   - Save current image")
        print("h   - Show help")
        print("+ / - - Increase/decrease exposure time")
        print("g / G - Decrease/increase gain")
        print("b / B - Decrease/increase binning")
        
        # Main loop
        while True:
            # Get frame from camera
            frame = camera.get_frame(5000)  # 5s timeout
            
            if frame is not None:
                # Calculate FPS
                frame_count += 1
                if frame_count >= 10:
                    end_time = time.time()
                    fps = frame_count / (end_time - start_time)
                    frame_count = 0
                    start_time = time.time()
                
                # Convert to 8-bit for OpenCV operations if needed
                bit_depth = camera.bit_depth
                if bit_depth > 8:
                    # Scale higher bit depth to 8-bit for display
                    gray = (frame >> (bit_depth - 8)).astype(np.uint8)
                    # Keep original for processing
                    original = frame.copy()
                else:
                    gray = frame.astype(np.uint8)
                    original = gray.copy()
                
                # Apply Laplacian filter
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                
                # Calculate RMS value of Laplacian (measure of image sharpness)
                rms = np.sqrt(np.mean(np.square(laplacian)))
                rms_values.append(rms)
                
                # Adjust max RMS value for better scaling if needed
                if rms > max_rms:
                    max_rms = rms * 1.2  # Add some headroom
                
                # Keep only the last 100 values for display
                if len(rms_values) > 100:
                    rms_values.pop(0)
                
                # Normalize Laplacian for display
                laplacian_display = cv2.normalize(laplacian, None, alpha=0, beta=255, 
                                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # Convert to BGR for display
                display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                # Get current temperature if cooler is available
                temp_str = ""
                if camera.has_cooler:
                    current_temp = camera.get_temperature()
                    if current_temp is not None:
                        temp_str = f"Temp: {current_temp:.1f}°C (Target: {args.cooler}°C)"
                
                # Display frames with info
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"RMS: {rms:.2f}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Exp: {current_exposure}ms", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Gain: {current_gain:.1f}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Bin: {args.bin}x{args.bin}", (10, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if temp_str:
                    cv2.putText(display_frame, temp_str, (10, 230), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.imshow('Mars Camera', display_frame)
                cv2.imshow('Laplacian Filter', laplacian_display)
                
                # Create RMS plot
                rms_plot = np.zeros((200, 400), dtype=np.uint8)
                for i in range(1, len(rms_values)):
                    y1 = 200 - int((rms_values[i-1] / max_rms) * 150)
                    y2 = 200 - int((rms_values[i] / max_rms) * 150)
                    # Clamp values to keep within range
                    y1 = max(0, min(199, y1))
                    y2 = max(0, min(199, y2))
                    x1 = (i-1) * 400 // max(1, len(rms_values)-1)
                    x2 = i * 400 // max(1, len(rms_values)-1)
                    cv2.line(rms_plot, (x1, y1), (x2, y2), 255, 1)
                
                # Draw horizontal line at current RMS value
                current_y = 200 - int((rms / max_rms) * 150)
                current_y = max(0, min(199, current_y))
                cv2.line(rms_plot, (0, current_y), (399, current_y), 128, 1)
                
                # Add scale to RMS plot
                cv2.putText(rms_plot, f"Max: {max_rms:.1f}", (300, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                
                cv2.imshow('RMS Trend', rms_plot)
            
            # Check for key press
            k = cv2.waitKey(1)
            if k == 27:  # ESC key
                break
            elif k == ord('s'):  # Save image
                # Create save directory if it doesn't exist
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                base_filename = os.path.join(args.save_dir, f"mars_capture_{timestamp}")
                
                # Save original bit depth image if available
                if bit_depth > 8 and 'original' in locals():
                    # For 16-bit PNG
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
                
                # Save Laplacian image
                cv2.imwrite(f"{base_filename}_laplacian.png", laplacian_display)
                print(f"Laplacian image saved as {base_filename}_laplacian.png")
                
            elif k == ord('h'):  # Help
                print("\nKeyboard controls:")
                print("ESC - Exit program")
                print("s   - Save current image")
                print("h   - Show this help")
                print("+ / - - Increase/decrease exposure time")
                print("g / G - Decrease/increase gain")
                print("b / B - Decrease/increase binning")
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
                new_bin = max(1, args.bin - 1)
                if new_bin != args.bin:
                    args.bin = new_bin
                    camera.stop_streaming()
                    camera.set_binning(args.bin)
                    camera.start_streaming()
                    print(f"Binning decreased to {args.bin}x{args.bin}")
            elif k == ord('B'):  # Increase binning
                new_bin = min(4, args.bin + 1)
                if new_bin != args.bin:
                    args.bin = new_bin
                    camera.stop_streaming()
                    camera.set_binning(args.bin)
                    camera.start_streaming()
                    print(f"Binning increased to {args.bin}x{args.bin}")
        
        # Clean up
        camera.disconnect()
        cv2.destroyAllWindows()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always ensure proper cleanup
        camera.disconnect()
        cv2.destroyAllWindows()
        print("Application terminated")

if __name__ == "__main__":
    main()