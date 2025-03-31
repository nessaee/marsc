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
    parser.add_argument('--exposure', type=int, help='Exposure time in milliseconds', default=100)
    parser.add_argument('--gain', type=float, help='Gain value (0-100)', default=0)
    parser.add_argument('--offset', type=int, help='Black level offset', default=0)
    parser.add_argument('--usb-limit', type=int, help='USB bandwidth limit (35-100%)', default=80)
    parser.add_argument('--bin', type=int, help='Binning (1, 2, 3, 4)', default=1, choices=[1, 2, 3, 4])
    parser.add_argument('--cooler', type=int, help='Cooler target temperature (Celsius)', default=29.1) #29.1

    parser.add_argument('--save-dir', help='Directory to save images', default='.')
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
        
        # Set USB bandwidth limit to reduce transfer errors
        if hasattr(camera, 'sdk') and hasattr(camera.sdk, 'dll'):
            from mars_camera_wrapper import POAConfig
            status = camera.sdk.dll.POASetConfig(camera.camera_id, POAConfig.POA_USB_BANDWIDTH_LIMIT, args.usb_limit, 0)
            print(f"USB bandwidth limit set to {args.usb_limit}%")
        
        # Enable cooler if available
        has_cooler = camera.has_cooler()
        if has_cooler:
            camera.set_target_temp(args.cooler)
            camera.cooler_on()
            print(f"Cooler enabled with target temperature {args.cooler}°C")
            print(f"Current temperature: {camera.get_temp():.1f}°C")
        else:
            print("This camera does not have a cooler")
        
        # Print camera specs
        print("\nCamera Specifications:")
        for key, value in camera.info.items():
            print(f"  {key}: {value}")
        
        # Start video capture
        camera.start_streaming()
        print("Camera streaming started")
        
        # Create single combined display window
        cv2.namedWindow('Mars Camera Combined View', cv2.WINDOW_NORMAL)
        
        # Check if camera is color or mono
        is_color = camera.info.get('is_color', False)
        print(f"Camera is {'color' if is_color else 'monochrome'}")
        
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
                
                # Detect and process image based on bit depth and actual data range
                bit_depth = camera.get_bit_depth()
                
                # Keep original for processing and saving
                original = frame.copy()
                
                # Use camera's reported bit depth for more accuracy
                actual_max = np.max(frame)
                hardware_bit_depth = bit_depth
                
                # But also try to detect actual bit depth from data
                detected_bit_depth = 0
                test_value = 255
                while test_value < actual_max and detected_bit_depth < 16:
                    detected_bit_depth += 1
                    test_value = (test_value << 1) + 1
                
                # Ensure minimum bit depth of 8 and cap at hardware bit depth
                detected_bit_depth = min(max(8, detected_bit_depth), 16)
                
                # For dynamic range percentage calculations, use hardware bit depth if valid
                bit_depth_for_calc = hardware_bit_depth if hardware_bit_depth > 0 else detected_bit_depth
                
                # Check if we have enough dynamic range to display
                frame_min = np.min(frame)
                frame_max = np.max(frame)
                dynamic_range = frame_max - frame_min
                
                # Create scaled version for OpenCV display
                # OpenCV requires 8-bit for display, but we'll preserve native bit depth for processing
                
                # New option: user can select display mode with 'd' key
                # Initialize display mode if not set
                if not hasattr(main, 'display_mode'):
                    main.display_mode = 2  # Default to native bit depth mode
                
                # Figure out the appropriate display method based on bit depth
                # Use the camera's known bit depth when available, otherwise use detected
                display_bit_depth = hardware_bit_depth if hardware_bit_depth > 0 else detected_bit_depth
                max_theoretical_val = float((1 << min(display_bit_depth, 24)) - 1)  # Max value for display bit depth
                
                if main.display_mode == 1:
                    # Full dynamic range stretching - uses actual min/max of current frame
                    if dynamic_range > 0:
                        gray = np.clip(((frame - frame_min) * 255.0 / dynamic_range), 0, 255).astype(np.uint8)
                    else:
                        gray = np.zeros_like(frame, dtype=np.uint8)
                    scaling_method = "Full dynamic range"
                    
                elif main.display_mode == 2:
                    # True native bit depth scaling - scales based on theoretical range of bit depth
                    # This shows exactly how the data appears in native bit depth
                    if detected_bit_depth <= 8:
                        # Direct conversion for 8-bit or less
                        gray = frame.astype(np.uint8)
                    else:
                        # Scale down to 8-bit preserving the entire bit depth range
                        scale_factor = 255.0 / max_theoretical_val
                        gray = np.clip(frame * scale_factor, 0, 255).astype(np.uint8)
                    scaling_method = "Native bit depth scale"
                    
                elif main.display_mode == 3:
                    # Bit shift only - preserves the relative values but loses lowest bits
                    if detected_bit_depth <= 8:
                        gray = frame.astype(np.uint8)
                        scaling_method = "Direct conversion"
                    else:
                        shift_amount = detected_bit_depth - 8
                        gray = (frame >> shift_amount).astype(np.uint8)
                        scaling_method = f"Bit shift (>>{shift_amount})"
                        
                else:
                    # Auto mode - chooses best method based on dynamic range of current frame
                    if dynamic_range > 50 and dynamic_range / max_theoretical_val > 0.1:
                        # Use histogram stretch if we have good dynamic range
                        gray = np.clip(((frame - frame_min) * 255.0 / dynamic_range), 0, 255).astype(np.uint8)
                        scaling_method = "Auto: Dynamic range"
                    else:
                        # Use native bit depth scaling otherwise
                        if detected_bit_depth <= 8:
                            gray = frame.astype(np.uint8)
                            scaling_method = "Auto: Direct conversion"
                        else:
                            scale_factor = 255.0 / max_theoretical_val
                            gray = np.clip(frame * scale_factor, 0, 255).astype(np.uint8)
                            scaling_method = "Auto: Native bit depth"
                
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
                
                # Before converting to BGR, check if the image is all white or very low contrast
                # If so, we'll try to enhance it
                if np.max(gray) - np.min(gray) < 20 or (np.mean(gray) > 240 and np.min(gray) > 200):
                    # Low contrast or all white image - try to enhance it
                    # Try CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    enhanced_gray = clahe.apply(gray)
                    
                    # Only use the enhanced version if it improved contrast
                    if np.max(enhanced_gray) - np.min(enhanced_gray) > np.max(gray) - np.min(gray):
                        gray = enhanced_gray
                        print("Applied CLAHE enhancement to low contrast image")
                
                # Prepare image for display based on camera type
                # If color camera and processing didn't already convert to RGB
                if is_color and len(gray.shape) < 3:
                    # Convert gray to RGB based on bayer pattern
                    bayer_pattern = camera.info.get('bayer_pattern', -1)
                    if bayer_pattern >= 0:  # Has a valid bayer pattern
                        # Convert bayer pattern to color
                        # OpenCV bayer pattern codes: 0=BG, 1=GB, 2=RG, 3=GR
                        # PlayerOne codes: 0=RG, 1=BG, 2=GR, 3=GB
                        cv_bayer_codes = {0: cv2.COLOR_BayerBG2BGR, 
                                         1: cv2.COLOR_BayerGB2BGR,
                                         2: cv2.COLOR_BayerRG2BGR, 
                                         3: cv2.COLOR_BayerGR2BGR}
                        display_frame = cv2.cvtColor(gray, cv_bayer_codes.get(bayer_pattern, cv2.COLOR_GRAY2BGR))
                    else:
                        # No valid bayer pattern, just convert to BGR
                        display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                else:
                    # Monochrome camera or already processed to RGB
                    if len(gray.shape) < 3:
                        display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    else:
                        display_frame = gray
                
                # Get current temperature if cooler is available
                temp_str = ""
                if has_cooler:
                    current_temp = camera.get_temp()
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
                cv2.putText(display_frame, f"Bin: {current_bin}x{current_bin}", (10, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display bit depth and scaling info
                cv2.putText(display_frame, f"Bit depth: {hardware_bit_depth}", (10, 230), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Scale: {scaling_method}", (10, 270), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # Display data range information
                # Get theoretical max value for the camera's true bit depth
                max_possible = float((1 << min(hardware_bit_depth, 24)) - 1)  # Prevent overflow with bit depths >24
                percent_of_max = min(100, int(100.0 * float(frame_max) / max_possible))
                cv2.putText(display_frame, f"Range: {frame_min}-{frame_max} ({dynamic_range})", (10, 310), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Usage: {percent_of_max}% of {hardware_bit_depth}-bit range", (10, 350), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                if temp_str:
                    cv2.putText(display_frame, temp_str, (10, 390), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Create a combined view with camera image, Laplacian, RMS plot, and info
                # Get sizes for proper layout
                h, w = display_frame.shape[:2]
                lap_h, lap_w = laplacian_display.shape[:2]
                
                # Convert laplacian to BGR for consistent display
                if len(laplacian_display.shape) < 3:
                    laplacian_display = cv2.cvtColor(laplacian_display, cv2.COLOR_GRAY2BGR)
                
                # Create RMS plot
                rms_plot_h, rms_plot_w = 200, 400
                rms_plot = np.zeros((rms_plot_h, rms_plot_w, 3), dtype=np.uint8)
                for i in range(1, len(rms_values)):
                    y1 = rms_plot_h - int((rms_values[i-1] / max_rms) * 150)
                    y2 = rms_plot_h - int((rms_values[i] / max_rms) * 150)
                    # Clamp values to keep within range
                    y1 = max(0, min(rms_plot_h-1, y1))
                    y2 = max(0, min(rms_plot_h-1, y2))
                    x1 = (i-1) * rms_plot_w // max(1, len(rms_values)-1)
                    x2 = i * rms_plot_w // max(1, len(rms_values)-1)
                    cv2.line(rms_plot, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                # Draw horizontal line at current RMS value
                current_y = rms_plot_h - int((rms / max_rms) * 150)
                current_y = max(0, min(rms_plot_h-1, current_y))
                cv2.line(rms_plot, (0, current_y), (rms_plot_w-1, current_y), (0, 128, 255), 1)
                cv2.putText(rms_plot, f"Max: {max_rms:.1f}", (rms_plot_w-100, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Create info display area - match height to the total combined view height
                info_w = 450
                # Calculate appropriate info panel height based on display scaling
                display_h_resized = int(h * scale_factor) if 'scale_factor' in locals() else h
                lap_h_resized = int(lap_h * scale_factor) if 'scale_factor' in locals() else lap_h
                # Use scaled heights for calculations
                info_h = max(display_h_resized, lap_h_resized + rms_plot_h)
                info_display = np.zeros((info_h, info_w, 3), dtype=np.uint8)
                
                # Draw divider line
                cv2.line(info_display, (0, 0), (0, info_h), (128, 128, 128), 1)
                
                # Display camera specifications
                y_pos = 30
                cv2.putText(info_display, f"Camera: {camera.get_name()}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_pos += 25
                
                cv2.putText(info_display, f"SN: {camera.get_sn()}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_pos += 25
                
                cv2.putText(info_display, f"Sensor: {camera.info.get('sensor', 'Unknown')}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_pos += 25
                
                # Resolution info
                max_res = f"{camera.info.get('max_width', 0)}x{camera.info.get('max_height', 0)}"
                curr_width, curr_height = camera.get_size()
                curr_res = f"{curr_width}x{curr_height}"
                cv2.putText(info_display, f"Resolution: {curr_res}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_pos += 25
                
                cv2.putText(info_display, f"Pixel: {camera.info.get('pixel_size', 0):.2f}μm", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_pos += 30
                
                # Current settings
                cv2.putText(info_display, f"Settings:", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
                y_pos += 25
                
                cv2.putText(info_display, f"Exp: {current_exposure:.1f}ms", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_pos += 25
                
                cv2.putText(info_display, f"Gain: {current_gain:.1f}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_pos += 25
                
                cv2.putText(info_display, f"Offset: {args.offset}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_pos += 25
                
                cv2.putText(info_display, f"Bin: {current_bin}x{current_bin}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_pos += 25
                
                cv2.putText(info_display, f"USB BW: {args.usb_limit}%", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_pos += 30
                
                # Image information
                cv2.putText(info_display, f"Image:", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
                y_pos += 25
                
                cv2.putText(info_display, f"Bit depth: {hardware_bit_depth}-bit", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_pos += 25
                
                cv2.putText(info_display, f"Scaling: {scaling_method}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_pos += 25
                
                # Stats 
                cv2.putText(info_display, f"Range: {frame_min}-{frame_max}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_pos += 25
                
                cv2.putText(info_display, f"DR usage: {percent_of_max}%", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_pos += 25
                
                cv2.putText(info_display, f"FPS: {fps:.1f}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_pos += 25
                
                cv2.putText(info_display, f"Focus RMS: {rms:.2f}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_pos += 25
                
                # Add temperature information if available
                if has_cooler and current_temp is not None:
                    cv2.putText(info_display, f"Temp: {current_temp:.1f}°C", 
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                    y_pos += 25
                
                # Add keyboard help near the bottom
                help_y = info_h - 110
                cv2.putText(info_display, "Controls:", (10, help_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                help_y += 20
                cv2.putText(info_display, "ESC: Exit | s: Save", (10, help_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                help_y += 20
                cv2.putText(info_display, "+/-: Exp | g/G: Gain", (10, help_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                help_y += 20
                cv2.putText(info_display, "b/B: Bin | d: Display", (10, help_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                help_y += 20
                cv2.putText(info_display, "h: Help", (10, help_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Combine all elements into a single display
                # Layout:
                # [Camera Image] [Info Panel]
                # [Laplacian  ] 
                # [RMS Plot   ]
                
                # Debug info - print frame stats
                f_min, f_mean, f_max = np.min(display_frame), np.mean(display_frame), np.max(display_frame)
                print(f"Frame stats: min={f_min:.1f}, mean={f_mean:.1f}, max={f_max:.1f}, range={f_max-f_min:.1f}   ", end="\r")
                
                # Auto-detect if contrast enhancement is needed and initialize contrast mode
                if not hasattr(main, 'contrast_boost'):
                    if np.min(display_frame) > 240 or np.max(display_frame) - np.min(display_frame) < 30:  
                        # If image appears all white or has very low contrast
                        main.contrast_boost = 1  # Start with enhanced contrast
                        print("Detected low contrast display, enabling enhanced contrast")
                    else:
                        main.contrast_boost = 0  # Normal contrast

                # Apply contrast enhancement if enabled
                if main.contrast_boost == 1:
                    # Enhanced contrast - stretch histogram to use full range
                    if len(display_frame.shape) == 3:
                        # Process each channel for RGB
                        for ch in range(3):
                            p_low, p_high = np.percentile(display_frame[:,:,ch], [5, 95])
                            if p_high > p_low:
                                display_frame[:,:,ch] = np.clip(255.0 * (display_frame[:,:,ch] - p_low) / (p_high - p_low), 0, 255).astype(np.uint8)
                    else:
                        # Process grayscale image
                        p_low, p_high = np.percentile(display_frame, [5, 95])
                        if p_high > p_low:
                            display_frame = np.clip(255.0 * (display_frame - p_low) / (p_high - p_low), 0, 255).astype(np.uint8)
                elif main.contrast_boost == 2:
                    # High contrast - use histogram equalization 
                    if len(gray.shape) < 3:  # Only for grayscale
                        # Apply histogram equalization
                        temp = cv2.equalizeHist(gray)
                        display_frame = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
                    elif len(display_frame.shape) == 3:
                        # For color images, equalize in YUV color space
                        yuv = cv2.cvtColor(display_frame, cv2.COLOR_BGR2YUV)
                        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])  # Equalize Y channel
                        display_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                
                # Create a combined view
                # For high-resolution cameras, resize everything for display
                scale_factor = 1.0
                if w > 1200:  # Large resolution camera
                    scale_factor = 1200.0 / w
                    
                # Apply scaling
                display_w = int(w * scale_factor)
                display_h = int(h * scale_factor)
                display_frame_resized = cv2.resize(display_frame, (display_w, display_h))
                
                # Resize laplacian too
                lap_w_resized = int(lap_w * scale_factor)
                lap_h_resized = int(lap_h * scale_factor)
                laplacian_display_resized = cv2.resize(laplacian_display, (lap_w_resized, lap_h_resized))
                
                # Calculate combined image size
                img_stack_h = lap_h_resized + rms_plot_h
                total_width = display_w + info_w
                total_height = max(display_h, img_stack_h)
                
                # Create combined image
                combined_view = np.zeros((total_height, total_width, 3), dtype=np.uint8)
                
                # Add main image
                combined_view[0:display_h, 0:display_w] = display_frame_resized
                
                # Add info panel - ensure the dimensions match
                info_h_to_use = min(info_h, total_height)
                info_display_resized = cv2.resize(info_display, (info_w, info_h_to_use))
                combined_view[0:info_h_to_use, display_w:display_w+info_w] = info_display_resized
                
                # Now add laplacian and RMS plot
                # Two possible layouts depending on aspect ratio
                if display_h < img_stack_h:
                    # Room for laplacian below the main image
                    lap_y_offset = display_h
                    # Ensure we're not exceeding the combined view size
                    lap_h_to_use = min(lap_h_resized, total_height - lap_y_offset)
                    if lap_h_to_use > 0:
                        combined_view[lap_y_offset:lap_y_offset+lap_h_to_use, 0:lap_w_resized] = laplacian_display_resized[:lap_h_to_use, :, :]
                    
                    # Add RMS plot below laplacian if there's room
                    rms_y_offset = lap_y_offset + lap_h_to_use
                    if rms_y_offset < total_height:
                        rms_h_to_use = min(rms_plot_h, total_height - rms_y_offset)
                        combined_view[rms_y_offset:rms_y_offset+rms_h_to_use, 0:rms_plot_w] = rms_plot[:rms_h_to_use, :, :]
                else:
                    # Not enough vertical space, add small versions to the side
                    # Resize even more for compact display
                    compact_scale = 0.3
                    small_lap_h = int(lap_h_resized * compact_scale)
                    small_lap_w = int(lap_w_resized * compact_scale)
                    small_lap = cv2.resize(laplacian_display_resized, (small_lap_w, small_lap_h))
                    
                    small_rms_h = int(rms_plot_h * compact_scale)
                    small_rms_w = int(rms_plot_w * compact_scale)
                    small_rms = cv2.resize(rms_plot, (small_rms_w, small_rms_h))
                    
                    # Position at bottom right of image panel
                    lap_x = display_w - small_lap_w - 10
                    lap_y = display_h - small_lap_h - small_rms_h - 10
                    if lap_y > 0 and lap_x > 0:
                        combined_view[lap_y:lap_y+small_lap_h, lap_x:lap_x+small_lap_w] = small_lap
                        combined_view[lap_y+small_lap_h+5:lap_y+small_lap_h+5+small_rms_h, 
                                    lap_x:lap_x+small_rms_w] = small_rms
                
                # Show the combined view
                cv2.imshow('Mars Camera Combined View', combined_view)
                
                # Old RMS plot code removed - now part of combined view
            
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
                print("d   - Cycle through display modes (Auto/Dynamic Range/Native/Bit-shift)")
                print("c   - Cycle through contrast modes (Normal/Enhanced/High)")
            elif k == ord('d'):  # Change display mode
                if not hasattr(main, 'display_mode'):
                    main.display_mode = 0
                main.display_mode = (main.display_mode + 1) % 4
                mode_names = ["Auto", "Full dynamic range", "Native bit depth", "Bit shift"]
                print(f"Display mode changed to: {mode_names[main.display_mode]}")
            elif k == ord('c'):  # Change contrast enhancement
                if not hasattr(main, 'contrast_boost'):
                    main.contrast_boost = 0
                main.contrast_boost = (main.contrast_boost + 1) % 3
                contrast_modes = ["Normal", "Enhanced", "High contrast"]
                print(f"Contrast mode changed to: {contrast_modes[main.contrast_boost]}")
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
            if has_cooler and camera.is_cooler_on():
                camera.cooler_off()
            camera.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()