#!/usr/bin/env python3

import cv2
import numpy as np
import time
import argparse
import sys
import os
import platform
from mars_camera_wrapper import POACamera

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
    parser.add_argument('--width', type=int, help='Display width', default=1600)
    parser.add_argument('--height', type=int, help='Display height', default=900)
    args = parser.parse_args()

    try:
        # List available cameras
        cameras = POACamera.get_cameras()
        if not cameras:
            print("No Player One cameras found. Please check your connection.")
            print("You may need to install udev rules for the camera device.")
            sys.exit(1)
        
        print(f"Found {len(cameras)} camera(s):")
        for i, cam in enumerate(cameras):
            print(f"[{i}] {cam['name']} (S/N: {cam['sn']})")
        
        # Initialize camera
        if args.sn:
            camera = POACamera(serial_number=args.sn)
        else:
            idx = min(args.index, len(cameras)-1)
            camera = POACamera(serial_number=cameras[idx]['sn'])
        
        # Open camera
        if not camera.open():
            print("Failed to open camera")
            sys.exit(1)
        
        print(f"Connected to: {camera.get_name()} (S/N: {camera.get_sn()})")
        
        # Check if it's a Mars series camera
        if not "Mars" in camera.get_name():
            print("Warning: This may not be a Mars camera. Script optimized for Mars-C and Mars-M II.")
        
        # Configure camera settings
        camera.set_binning(args.bin)
        width, height = camera.get_size()
        print(f"Resolution: {width}x{height}, {camera.get_bit_depth()}-bit")
        
        camera.set_roi(0, 0, width, height)  # Full frame
        camera.set_exposure(args.exposure)
        print(f"Exposure set to {args.exposure}ms")
        
        camera.set_gain(args.gain)
        print(f"Gain set to {args.gain}")
        
        camera.set_offset(args.offset)
        print(f"Offset set to {args.offset}")
        
        # Enable cooler if available
        has_cooler = camera.has_cooler()
        if has_cooler:
            camera.set_target_temp(args.cooler)
            camera.cooler_on()
            print(f"Cooler enabled with target temperature {args.cooler}°C")
            print(f"Current temperature: {camera.get_temp():.1f}°C")
        else:
            print("This camera does not have a cooler")
        
        # Start video capture
        camera.start_streaming()
        print("Camera streaming started")
        
        # Create a single combined display window
        cv2.namedWindow('Mars Camera Display', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Mars Camera Display', args.width, args.height)
        
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
        current_bin = args.bin
        
        print("\nStreaming... Keyboard controls:")
        print("ESC - Exit program")
        print("s   - Save current image")
        print("h   - Show help")
        print("+ / - - Increase/decrease exposure time")
        print("g / G - Decrease/increase gain")
        print("b / B - Decrease/increase binning")
        print("r   - Reset auto-scaling for RMS display")
        print("a   - Toggle auto-stretch contrast")
        
        # Flag to track if we've had at least one successful frame
        had_frame = False
        
        # Auto stretch for better visualization
        auto_stretch = True
        stretch_min = 0
        stretch_max = 255
        
        # Main loop
        while True:
            # Get frame from camera
            frame = camera.get_video_data(1000)  # 1s timeout
            
            if frame is not None:
                had_frame = True
                
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
                    # Keep original for processing and saving
                    original = frame.copy()
                    
                    # Auto-stretch for display if enabled
                    if auto_stretch and frame.size > 0:
                        # Compute 5-95 percentile for better contrast
                        p_low = np.percentile(frame, 5)
                        p_high = np.percentile(frame, 95)
                        # Gradually update stretch parameters for smooth transitions
                        stretch_min = 0.9 * stretch_min + 0.1 * p_low
                        stretch_max = 0.9 * stretch_max + 0.1 * p_high
                        
                        # Apply stretch with clipping
                        frame_stretched = np.clip((frame - stretch_min) * 255.0 / (stretch_max - stretch_min), 0, 255).astype(np.uint8)
                        gray = frame_stretched
                    else:
                        # Simple bit shift for linear conversion to 8-bit
                        gray = (frame >> (bit_depth - 8)).astype(np.uint8)
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
                
                # Ensure we have RGB displays for combined view
                if len(gray.shape) == 2:
                    display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                else:
                    display_frame = gray.copy()
                
                laplacian_display_color = cv2.cvtColor(laplacian_display, cv2.COLOR_GRAY2BGR)
                
                # Add info to display frame
                # Get current temperature if cooler is available
                temp_str = ""
                if has_cooler:
                    current_temp = camera.get_temp()
                    if current_temp is not None:
                        temp_str = f"Temp: {current_temp:.1f}°C (Target: {args.cooler}°C)"
                
                # Create info bar with stats
                info_height = 60
                info_bar = np.zeros((info_height, display_frame.shape[1] * 2, 3), dtype=np.uint8)
                
                # Add text with stats
                text_y = 25
                camera_name = camera.get_name()
                cv2.putText(info_bar, f"{camera_name} | FPS: {fps:.1f} | RMS: {rms:.2f} | Exp: {current_exposure}ms | Gain: {current_gain:.1f} | Bin: {current_bin}x{current_bin}", 
                           (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if temp_str:
                    cv2.putText(info_bar, temp_str, (10, text_y + 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Create RMS trend plot
                rms_plot_height = 150
                rms_plot_width = display_frame.shape[1] * 2
                rms_plot = np.zeros((rms_plot_height, rms_plot_width, 3), dtype=np.uint8)
                
                # Draw grid lines
                for i in range(0, rms_plot_width, 50):
                    cv2.line(rms_plot, (i, 0), (i, rms_plot_height), (50, 50, 50), 1)
                for i in range(0, rms_plot_height, 25):
                    cv2.line(rms_plot, (0, i), (rms_plot_width, i), (50, 50, 50), 1)
                
                # Draw RMS trend
                if len(rms_values) > 1:
                    for i in range(1, len(rms_values)):
                        y1 = rms_plot_height - int((rms_values[i-1] / max_rms) * (rms_plot_height - 20))
                        y2 = rms_plot_height - int((rms_values[i] / max_rms) * (rms_plot_height - 20))
                        # Clamp values to stay within bounds
                        y1 = max(0, min(rms_plot_height-1, y1))
                        y2 = max(0, min(rms_plot_height-1, y2))
                        x1 = (i-1) * rms_plot_width // max(1, len(rms_values)-1)
                        x2 = i * rms_plot_width // max(1, len(rms_values)-1)
                        cv2.line(rms_plot, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw horizontal line at current RMS value
                current_y = rms_plot_height - int((rms / max_rms) * (rms_plot_height - 20))
                current_y = max(0, min(rms_plot_height-1, current_y))
                cv2.line(rms_plot, (0, current_y), (rms_plot_width-1, current_y), (255, 0, 0), 1)
                
                # Add scale to RMS plot
                cv2.putText(rms_plot, f"RMS Max: {max_rms:.1f}", (rms_plot_width-150, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add "Raw Image" label to display_frame
                cv2.putText(display_frame, "Raw Image", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Add "Laplacian Filter" label to laplacian_display_color
                cv2.putText(laplacian_display_color, "Laplacian Filter", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Ensure the display frames have the same height
                h1, w1 = display_frame.shape[:2]
                h2, w2 = laplacian_display_color.shape[:2]
                
                # Resize if necessary to make heights match
                if h1 != h2:
                    scale = h1 / h2
                    laplacian_display_color = cv2.resize(laplacian_display_color, (int(w2 * scale), h1))
                
                # Put images side by side
                combined_frame = np.hstack((display_frame, laplacian_display_color))
                
                # Add info bar and RMS plot
                full_display = np.vstack((info_bar, combined_frame, rms_plot))
                
                # Display the combined frame
                cv2.imshow('Mars Camera Display', full_display)
            elif not had_frame:
                # If we haven't received a frame yet, show a waiting message
                waiting_msg = np.zeros((400, 800, 3), dtype=np.uint8)
                cv2.putText(waiting_msg, "Waiting for camera data...", (100, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Mars Camera Display', waiting_msg)
            
            # Check for key press
            k = cv2.waitKey(1)
            if k == 27:  # ESC key
                break
            elif k == ord('s'):  # Save image
                # Create save directory if it doesn't exist
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                
                if not had_frame:
                    print("No frame to save yet!")
                    continue
                
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
                
                # Save combined display
                cv2.imwrite(f"{base_filename}_display.jpg", full_display)
                print(f"Display view saved as {base_filename}_display.jpg")
                
            elif k == ord('h'):  # Help
                print("\nKeyboard controls:")
                print("ESC - Exit program")
                print("s   - Save current image")
                print("h   - Show this help")
                print("+ / - - Increase/decrease exposure time")
                print("g / G - Decrease/increase gain")
                print("b / B - Decrease/increase binning")
                print("r   - Reset auto-scaling for RMS display")
                print("a   - Toggle auto-stretch contrast")
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
            elif k == ord('r'):  # Reset RMS plot scaling
                max_rms = 20
                rms_values = []
                print("RMS plot scaling reset")
            elif k == ord('a'):  # Toggle auto-stretch
                auto_stretch = not auto_stretch
                print(f"Auto-stretch {'enabled' if auto_stretch else 'disabled'}")
        
        # Clean up
        print("Stopping camera...")
        camera.stop_streaming()
        
        # Turn off cooler if it was enabled
        if has_cooler:
            camera.cooler_off()
            print("Cooler turned off")
        
        camera.close()
        cv2.destroyAllWindows()
        print("Application terminated")
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always ensure proper cleanup
        if 'camera' in locals() and camera.is_connected():
            if camera.is_streaming():
                camera.stop_streaming()
            if 'has_cooler' in locals() and has_cooler and camera.is_cooler_on():
                camera.cooler_off()
            camera.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()