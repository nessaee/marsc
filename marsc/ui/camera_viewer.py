#!/usr/bin/env python3

import cv2
import numpy as np
import time
import queue
import threading
import os

class CameraViewer:
    """Handles camera viewing, controls, and user interaction"""
    
    def __init__(self, camera, image_processor, window_name="Mars Camera"):
        self.camera = camera
        self.processor = image_processor
        self.window_name = window_name
        self.running = False
        self.show_laplacian = True  # Flag to show Laplacian view
        
        # Setup single display window with WINDOW_NORMAL to allow manual resizing
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Current settings for interactive control
        self.current_exposure = None  # Will be set in start()
        self.current_gain = None      # Will be set in start()
        # Fixed resolution of 1944x1096 - no binning needed
        
        # Performance metrics
        self.processing_fps = 0
        self.acquisition_fps = 0
        
        # Setup frame queue for multithreading
        self.frame_queue = queue.Queue(maxsize=5)  # Limit queue size to prevent memory issues
        
    def _exposure_callback(self, value):
        # Trackbar returns int 0-1000, convert to exposure in ms (0.1ms - 5000ms)
        # Use logarithmic scale for better control
        if value == 0:
            value = 1  # Prevent log(0)
            
        # Map 1-1000 to 0.1-5000ms (logarithmic)
        exp_ms = 0.1 * (10 ** (3 * value / 1000))
        self.current_exposure = exp_ms
        self.camera.set_exposure(exp_ms)
        
    def _gain_callback(self, value):
        # Trackbar returns 0-100 directly for gain
        self.current_gain = value
        self.camera.set_gain(value)
        
    # Removed binning functionality as we now use fixed resolution
    
    def _setup_control_panel(self):
        """Create control panel with trackbars"""
        # Create trackbars directly on the main window
        # For exposure, map 0-1000 to 0.1-5000ms logarithmically
        # Convert current exposure to trackbar value
        if self.current_exposure <= 0.1:
            exp_trackbar = 0
        else:
            # Inverse of the mapping function in _exposure_callback
            exp_trackbar = int(1000 * np.log10(self.current_exposure / 0.1) / 3)
            exp_trackbar = max(0, min(1000, exp_trackbar))
            
        cv2.createTrackbar("Exposure (0.1-5000ms)", self.window_name, 
                         exp_trackbar, 1000, self._exposure_callback)
        
        # For gain, direct mapping 0-100
        gain_trackbar = int(max(0, min(100, self.current_gain)))
        cv2.createTrackbar("Gain (0-100)", self.window_name, 
                         gain_trackbar, 100, self._gain_callback)
        
        # Fixed resolution of 1944x1096 - no binning trackbar needed
        
    def print_help(self):
        """Print help information"""
        print("\nKeyboard controls:")
        print("ESC - Exit program")
        print("s   - Save current image")
        print("h   - Show this help")
        print("+ / - - Increase/decrease exposure time")
        print("[ / ] - Decrease/increase gain")
        # Fixed resolution - no binning adjustment needed
        print("d   - Cycle through display modes (Auto/Dynamic Range/Native/Bit-shift)")
        print("c   - Cycle through contrast modes (Normal/Enhanced/High)")
        print("g   - Cycle through grid modes (None/Rule of Thirds/Golden Ratio/Fine Grid)")
        print("r   - Reset RMS plot scaling")
        print("p   - Toggle real-time RMS plot window")
        print("l   - Toggle Laplacian filter view")
        
    def start(self, initial_settings=None):
        """Start the viewer with initial settings"""
        if not self.camera.connected:
            print("Camera not connected")
            return False
            
        # Apply initial settings if provided
        if initial_settings:
            if 'exposure' in initial_settings:
                self.current_exposure = initial_settings['exposure']
                self.camera.set_exposure(self.current_exposure)
                
            if 'gain' in initial_settings:
                self.current_gain = initial_settings['gain']
                self.camera.set_gain(self.current_gain)
                
            # Fixed resolution of 1944x1096 - no binning configuration needed
                
            # Apply other settings
            if 'offset' in initial_settings:
                self.camera.set_offset(initial_settings['offset'])
                
            if 'usb_limit' in initial_settings:
                self.camera.set_usb_limit(initial_settings['usb_limit'])
                
            if 'cooler_temp' in initial_settings and self.camera.has_cooler():
                self.camera.set_target_temp(initial_settings['cooler_temp'])
                self.camera.cooler_on()
                print(f"Cooler enabled, target: {initial_settings['cooler_temp']}°C, current: {self.camera.get_temp():.1f}°C")
                
        # Make sure we have current values
        if self.current_exposure is None:
            self.current_exposure = self.camera.get_exposure()
        if self.current_gain is None:
            self.current_gain = self.camera.get_gain()
        
        # Set up the embedded control panel
        self._setup_control_panel()
            
        # Start streaming
        if not self.camera.start_streaming():
            print("Failed to start streaming")
            return False
            
        self.print_help()
        self.running = True
        
        # Start frame acquisition thread
        acq_thread = threading.Thread(target=self._frame_acquisition_thread, daemon=True)
        acq_thread.start()
        
        return True
        
    def _frame_acquisition_thread(self):
        """Frame acquisition thread function"""
        acq_frame_count = 0
        acq_start_time = time.time()
        
        while self.running:
            # Get frame
            frame = self.camera.get_frame(timeout_ms=500)  # Reduced timeout for faster response
            if frame is not None:
                # If queue is full, remove oldest frame to make room
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # Add new frame to queue
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
                
                # Calculate acquisition FPS
                acq_frame_count += 1
                if acq_frame_count >= 10:
                    acq_end_time = time.time()
                    self.acquisition_fps = acq_frame_count / (acq_end_time - acq_start_time)
                    acq_frame_count = 0
                    acq_start_time = time.time()
    
    def run(self, save_dir="."):
        """Main processing and display loop"""
        if not self.running:
            return
            
        # Setup FPS calculation
        frame_count = 0
        start_time = time.time()
        
        # Main loop
        last_processed_data = None
        
        while self.running:
            # Get frame from queue
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                # No frame available, check for key presses and continue
                k = cv2.waitKey(1)
                if k == 27:  # ESC key
                    self.running = False
                    break
                continue
            
            # Calculate processing FPS
            frame_count += 1
            if frame_count >= 10:
                end_time = time.time()
                self.processing_fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Process the frame
            processed_data = self.processor.process_frame(frame, self.camera.get_bit_depth())
            if processed_data is None:
                continue
                
            # Keep track of the processed data for saving
            last_processed_data = processed_data
            
            # Allow manual window resizing by the user, no automatic adjustments
            
            # Create camera info dict for overlay
            camera_info = {
                'exposure': self.current_exposure,
                'gain': self.current_gain,
                'resolution': '1944x1096',  # Fixed resolution
                'rms': processed_data['rms'],
                'frame_min': processed_data['frame_min'],
                'frame_max': processed_data['frame_max'],
                'percent_of_max': processed_data['percent_of_max'],
                'scaling_method': processed_data['scaling_method'],
                'contrast_method': processed_data['contrast_method']
            }
            
            # Add temperature if camera has cooler
            if self.camera.has_cooler():
                camera_info['temperature'] = self.camera.get_temp()
                
            # FPS info
            fps_info = {
                'display_fps': self.processing_fps,
                'camera_fps': self.acquisition_fps
            }
            
            # Add info overlay to frame
            display_frame = self.processor.add_info_overlay(
                processed_data['display'], camera_info, fps_info)
            
            # Create a combined display with all elements
            combined_display = self._create_combined_display(display_frame, processed_data, camera_info)
            
            # Display the consolidated view
            cv2.imshow(self.window_name, combined_display)
            
            # Check for key press and handle commands
            k = cv2.waitKey(1)
            if k == 27:  # ESC key
                self.running = False
                break
            elif k == ord('s'):  # Save image
                if last_processed_data:
                    # Create timestamp
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    base_filename = os.path.join(save_dir, f"mars_capture_{timestamp}")
                    self.processor.save_images(last_processed_data, base_filename)
            elif k == ord('h'):  # Help
                self.print_help()
            elif k == ord('d'):  # Change display mode
                mode_name = self.processor.cycle_display_mode()
                print(f"Display mode changed to: {mode_name}")
            elif k == ord('c'):  # Change contrast mode
                contrast_name = self.processor.cycle_contrast_mode()
                print(f"Contrast mode changed to: {contrast_name}")
            elif k == ord('g'):  # Change grid mode
                grid_name = self.processor.cycle_grid_mode()
                print(f"Grid mode changed to: {grid_name}")
            elif k == ord('r'):  # Reset RMS plot scaling
                if self.processor.reset_rms_plot_scale():
                    print("RMS plot scaling reset")
                else:
                    print("RMS plot not active")
            elif k == ord('p'):  # Toggle RMS plot
                is_visible = self.processor.toggle_rms_plot()
                print(f"RMS plot {'enabled' if is_visible else 'disabled'}")
            elif k == ord('+') or k == ord('='):  # Increase exposure
                self.current_exposure = min(5000, self.current_exposure * 1.2)
                self.camera.set_exposure(self.current_exposure)
                print(f"Exposure increased to {self.current_exposure:.1f}ms")
            elif k == ord('-'):  # Decrease exposure
                self.current_exposure = max(0.1, self.current_exposure / 1.2)
                self.camera.set_exposure(self.current_exposure)
                print(f"Exposure decreased to {self.current_exposure:.1f}ms")
            elif k == ord('['):  # Decrease gain
                self.current_gain = max(0, self.current_gain - 1)
                self.camera.set_gain(self.current_gain)
                print(f"Gain decreased to {self.current_gain:.1f}")
            elif k == ord(']'):  # Increase gain
                self.current_gain = min(100, self.current_gain + 1)
                self.camera.set_gain(self.current_gain)
                print(f"Gain increased to {self.current_gain:.1f}")
            # Removed binning adjustment code as we now use fixed resolution
            elif k == ord('l'):  # Toggle Laplacian view
                self.show_laplacian = not self.show_laplacian
                print(f"Laplacian view {'enabled' if self.show_laplacian else 'disabled'}")
                
                # Let the user manually adjust window size as needed
            elif k == ord('m'):  # Toggle color mode (monochrome/color)
                mode_name = self.processor.toggle_color_mode()
                print(f"Display mode changed to: {mode_name}")
        
        # Clean up
        self.stop()
    
    def _create_combined_display(self, main_frame, processed_data, camera_info):
        """Create a combined display with main image, Laplacian, and controls"""
        # Get dimensions of main frame
        h, w = main_frame.shape[:2]
        
        # We'll create the control panel after determining the final width of the combined image
        
        # Create side panel for Laplacian if available
        if self.show_laplacian and 'laplacian_vis' in processed_data:
            # Get the Laplacian visualization
            laplacian_vis = processed_data['laplacian_vis']
            
            # Simple approach - resize both to match for clean side-by-side display
            # This prevents any artifacts from uneven scaling or padding
            # Make both the same size as the main frame to maintain proportions
            h, w = main_frame.shape[:2]
            
            # Resize the Laplacian to match main frame exactly
            laplacian_resized = cv2.resize(laplacian_vis, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Simple horizontal stack
            combined_horizontal = np.hstack((main_frame, laplacian_resized))
        else:
            combined_horizontal = main_frame
            
        # Create appropriately sized control panel to match the width of combined_horizontal
        control_img = np.zeros((200, combined_horizontal.shape[1], 3), dtype=np.uint8)
        
        # Display title and controls
        cv2.putText(control_img, "Mars Camera Controls", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display current values
        cv2.putText(control_img, f"Exposure: {camera_info['exposure']:.2f} ms", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(control_img, f"Gain: {camera_info['gain']:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(control_img, f"Resolution: {camera_info['resolution']}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(control_img, f"Focus score: {camera_info['rms']:.2f}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Stack the image area with the control panel vertically
        combined_display = np.vstack((combined_horizontal, control_img))
        
        return combined_display
    
    def stop(self):
        """Stop the viewer and clean up resources"""
        self.running = False
        
        # Stop camera streaming
        if self.camera.streaming:
            self.camera.stop_streaming()
            
        # Close all windows
        cv2.destroyAllWindows()
