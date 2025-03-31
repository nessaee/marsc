#!/usr/bin/env python3

import cv2
import numpy as np
import time
import argparse
import sys
import os
import platform

# Set the path directly to the SDK library folder based on architecture
script_dir = os.path.dirname(os.path.abspath(__file__))
sdk_dir = os.path.join(script_dir, "playerone_sdk/PlayerOne_Camera_SDK_Linux_V3.8.1")

# Determine architecture for selecting the right library
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

# Full path to the library
lib_dir = os.path.join(sdk_dir, "lib", lib_arch_dir)
lib_path = os.path.join(lib_dir, "libPlayerOneCamera.so")

# Check if the library exists
if not os.path.exists(lib_path):
    print(f"ERROR: Player One camera library not found at {lib_path}")
    print("Please make sure the SDK is properly extracted")
    sys.exit(1)

# Set environment variable for library path
os.environ['LD_LIBRARY_PATH'] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ['PLAYERONE_LIB_PATH'] = lib_path

print(f"Using library: {lib_path}")

# Import the pyPOACamera module
try:
    # Add SDK's python directory to path to find the module
    python_dir = os.path.join(sdk_dir, "python")
    sys.path.insert(0, python_dir)
    from pyPOACamera import POACamera
    print("Successfully imported pyPOACamera module")
except ImportError as e:
    print(f"Error importing pyPOACamera: {e}")
    print(f"Looking for the module in {python_dir}")
    print("Make sure the SDK is properly extracted")
    sys.exit(1)
except Exception as e:
    print(f"Error loading Player One camera library: {e}")
    print(f"Library path: {lib_path}")
    print("\nPossible solutions:")
    print("1. Make sure the Player One SDK is properly extracted")
    print("2. Install required dependencies: sudo apt-get install libusb-1.0-0")
    print("3. Run the following to install udev rules for camera access:")
    print(f"   sudo cp {os.path.join(sdk_dir, 'udev', '99-player_one_astronomy.rules')} /etc/udev/rules.d/")
    print("   sudo udevadm control --reload-rules && sudo udevadm trigger")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Stream from Player One Mars-C or Mars-M II sensor with Laplacian filtering')
    parser.add_argument('--index', type=int, help='Camera index (0-based)', default=0)
    parser.add_argument('--exposure', type=int, help='Exposure time in milliseconds', default=10)
    parser.add_argument('--gain', type=float, help='Gain value (0-100)', default=20.0)
    parser.add_argument('--offset', type=int, help='Black level offset', default=10)
    parser.add_argument('--bin', type=int, help='Binning (1, 2, 3, 4)', default=1, choices=[1, 2, 3, 4])
    parser.add_argument('--cooler', type=int, help='Cooler target temperature (Celsius)', default=-10)
    args = parser.parse_args()

    try:
        # Initialize the SDK and enumerate cameras
        print("Initializing Player One camera SDK...")
        cameras = POACamera.get_cameras()
        if not cameras:
            print("No Player One cameras found. Please check your connection and permissions.")
            print("You may need to install udev rules with the following commands:")
            print(f"sudo cp {os.path.join(sdk_dir, 'udev', '99-player_one_astronomy.rules')} /etc/udev/rules.d/")
            print("sudo udevadm control --reload-rules && sudo udevadm trigger")
            print("Then reconnect your camera and try again.")
            sys.exit(1)
        
        print(f"Found {len(cameras)} camera(s):")
        for i, cam in enumerate(cameras):
            print(f"[{i}] {cam.get('name')} (S/N: {cam.get('sn')})")
        
        # Select camera
        cam_idx = args.index
        if cam_idx >= len(cameras):
            print(f"Camera index {cam_idx} out of range. Using first camera (index 0).")
            cam_idx = 0
        
        camera_info = cameras[cam_idx]
        print(f"Selected: {camera_info.get('name')} (S/N: {camera_info.get('sn')})")
        
        # Check if it's a Mars series camera
        if not ("Mars" in camera_info.get('name')):
            print("Warning: This may not be a Mars camera. Script optimized for Mars-C and Mars-M II.")
        
        # Initialize the camera
        camera = POACamera(camera_info.get('sn'))
        
        # Connect to the camera
        if not camera.connect():
            raise Exception("Failed to connect to camera")
        
        print(f"Connected to camera: {camera.get_name()}")
        
        # Configure camera settings
        camera.set_binning(args.bin)
        print(f"Binning set to {args.bin}x{args.bin}")
        
        # Get resolution after binning
        width, height = camera.get_size()
        print(f"Image size: {width}x{height}")
        
        # Set ROI to full frame
        camera.set_roi(0, 0, width, height)
        
        # Configure exposure in milliseconds
        camera.set_exposure(args.exposure)
        print(f"Exposure set to {args.exposure}ms")
        
        # Set gain
        camera.set_gain(args.gain)
        print(f"Gain set to {args.gain}")
        
        # Set offset
        camera.set_offset(args.offset)
        print(f"Offset set to {args.offset}")
        
        # Enable cooler if available
        has_cooler = camera.has_cooler()
        if has_cooler:
            camera.set_target_temp(args.cooler)
            camera.cooler_on()
            print(f"Cooler enabled with target temperature {args.cooler}°C")
        else:
            print("This camera does not have a cooler")
        
        # Start video mode
        camera.start_video()
        print("Camera streaming started in video mode")
        
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
        
        print("\nStreaming... Keyboard controls:")
        print("ESC - Exit program")
        print("s   - Save current image")
        print("h   - Show help")
        print("+ / - - Increase/decrease exposure time")
        print("g / G - Decrease/increase gain")
        
        # Current settings for interactive control
        current_exposure = args.exposure
        current_gain = args.gain
        
        # Main loop
        while True:
            # Get frame from camera
            frame = camera.get_video_data(5000)  # 5s timeout
            
            if frame is not None:
                # Calculate FPS
                frame_count += 1
                if frame_count >= 10:
                    end_time = time.time()
                    fps = frame_count / (end_time - start_time)
                    frame_count = 0
                    start_time = time.time()
                
                # Convert to 8-bit for OpenCV operations if needed
                bit_depth = camera.get_bit_depth()
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
                if has_cooler:
                    current_temp = camera.get_temp()
                    target_temp = camera.get_target_temp()
                    temp_str = f"Temp: {current_temp:.1f}°C (Target: {target_temp}°C)"
                
                # Display frames with info
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"RMS: {rms:.2f}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Exp: {current_exposure}ms", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Gain: {current_gain:.1f}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if temp_str:
                    cv2.putText(display_frame, temp_str, (10, 190), 
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
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                # Save original bit depth image if available
                if bit_depth > 8 and 'original' in locals():
                    # For 16-bit PNG
                    if bit_depth <= 16:
                        cv2.imwrite(f"mars_capture_{timestamp}.png", original)
                        print(f"Full bit depth image saved as mars_capture_{timestamp}.png")
                
                # Save 8-bit image
                cv2.imwrite(f"mars_capture_{timestamp}_8bit.png", gray)
                print(f"8-bit image saved as mars_capture_{timestamp}_8bit.png")
                
                # Save stretched version for better visibility
                stretch_img = cv2.normalize(gray, None, alpha=0, beta=255, 
                                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(f"mars_capture_{timestamp}_stretched.png", stretch_img)
                print(f"Enhanced image saved as mars_capture_{timestamp}_stretched.png")
            elif k == ord('h'):  # Help
                print("\nKeyboard controls:")
                print("ESC - Exit program")
                print("s   - Save current image")
                print("h   - Show this help")
                print("+ / - - Increase/decrease exposure time")
                print("g / G - Decrease/increase gain")
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
        
        # Clean up
        print("Stopping camera...")
        camera.stop_video()
        
        # Turn off cooler if it was enabled
        if has_cooler:
            camera.cooler_off()
            print("Cooler turned off")
        
        # Disconnect
        camera.disconnect()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to clean up in case of error
        try:
            if 'camera' in locals() and camera is not None:
                if camera.is_connected():
                    camera.stop_video()
                    if 'has_cooler' in locals() and has_cooler:
                        camera.cooler_off()
                    camera.disconnect()
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    main()