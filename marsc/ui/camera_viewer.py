#!/usr/bin/env python3

import cv2
import numpy as np
import time
import queue
import threading
import os
from enum import Enum

class DisplayMode(Enum):
    AUTO = 0
    FULL_RANGE = 1
    NATIVE = 2
    BIT_SHIFT = 3

class CameraViewer:
    """Handles camera viewing, controls, and user interaction with modern GUI elements"""
    
    def __init__(self, camera, image_processor, window_name="Mars Camera"):
        self.camera = camera
        self.processor = image_processor
        self.window_name = window_name
        self.running = False
        self.show_laplacian = True  # Flag to show Laplacian view
        self.control_panel_width = 300  # Width of side control panel
        
        # Get screen dimensions
        self.detect_screen_size()
        
        # Setup main display window with WINDOW_NORMAL to allow manual resizing
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Set initial window size proportional to screen size (75% of screen)
        win_width = min(1920, int(self.screen_width * 0.75))
        win_height = min(1080, int(self.screen_height * 0.75))
        cv2.resizeWindow(self.window_name, win_width, win_height)
        
        # Create control panel window
        self.control_panel_name = "Control Panel"
        cv2.namedWindow(self.control_panel_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.control_panel_name, self.control_panel_width, 600)
        
        # Position the control panel window to the right of the main window
        cv2.moveWindow(self.control_panel_name, win_width + 30, 50)
        
        # Current settings for interactive control
        self.current_exposure = None  # Will be set in start()
        self.current_gain = None      # Will be set in start()
        # Fixed resolution of 1944x1096 - no binning needed
        
        # Performance metrics
        self.processing_fps = 0
        self.acquisition_fps = 0
        
        # Setup frame queue for multithreading
        self.frame_queue = queue.Queue(maxsize=5)  # Limit queue size to prevent memory issues
        
        # UI Colors
        self.BUTTON_COLOR = (70, 70, 80)
        self.BUTTON_HOVER_COLOR = (100, 100, 120)
        self.BUTTON_TEXT_COLOR = (255, 255, 255)
        self.BUTTON_ACTIVE_COLOR = (0, 140, 255)
        self.PANEL_COLOR = (50, 50, 55)
        self.HELP_BG_COLOR = (30, 30, 35, 200)  # with alpha
        
        # Track mouse position for UI interaction
        self.mouse_x = 0
        self.mouse_y = 0
        self.control_mouse_x = 0
        self.control_mouse_y = 0
        self.buttons = []  # List to store buttons for main window
        self.control_buttons = []  # List to store buttons for control panel
        self.show_help_menu = False  # Toggle for help menu display
        
        # Create button images and cache
        self.button_cache = {}
        
        # Set up mouse callbacks for button interaction
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        cv2.setMouseCallback(self.control_panel_name, self._control_mouse_callback)
    
    def detect_screen_size(self):
        """Detect screen resolution"""
        try:
            # Default fallback values if detection fails
            self.screen_width = 1920
            self.screen_height = 1080
            
            # Try to get actual screen resolution
            # This is a simple approach - works on Linux with X11
            try:
                import subprocess
                output = subprocess.check_output('xrandr | grep "\*" | cut -d" " -f4', shell=True).decode().strip()
                if 'x' in output:
                    width, height = output.split('x')
                    self.screen_width = int(width)
                    self.screen_height = int(height)
            except:
                # Fallback to a more conservative size on failure
                pass
        except Exception as e:
            print(f"Error detecting screen size: {e}")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for main window GUI interaction"""
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if help menu is open, clicking anywhere closes it
            if self.show_help_menu:
                self.show_help_menu = False
                return
                
            # Check if any button was clicked
            for button in self.buttons:
                if self._is_point_in_rect(x, y, button['rect']):
                    # Call the button's action function
                    if button['action'] is not None:
                        button['action']()
                    break
    
    def _control_mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for control panel GUI interaction"""
        self.control_mouse_x = x
        self.control_mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if any control panel button was clicked
            for button in self.control_buttons:
                if self._is_point_in_rect(x, y, button['rect']):
                    # Call the button's action function
                    if button['action'] is not None:
                        button['action']()
                    break
    
    def _is_point_in_rect(self, x, y, rect):
        """Check if point (x,y) is inside rectangle (x, y, width, height)"""
        return (rect[0] <= x <= rect[0] + rect[2] and 
                rect[1] <= y <= rect[1] + rect[3])
    
    def _create_button(self, label, x, y, width, height, action=None, active=False, tooltip=None, panel="main"):
        """Create a button with the given properties"""
        button = {
            'label': label,
            'rect': (x, y, width, height),
            'action': action,
            'active': active,
            'tooltip': tooltip,
            'hover': False
        }
        
        # Add to appropriate button list
        if panel == "control":
            self.control_buttons.append(button)
        else:
            self.buttons.append(button)
            
        return button
    
    def _draw_button(self, img, button, is_control_panel=False):
        """Draw a button on the image"""
        x, y, w, h = button['rect']
        label = button['label']
        
        # Check if mouse is hovering over button
        if is_control_panel:
            button['hover'] = self._is_point_in_rect(self.control_mouse_x, self.control_mouse_y, button['rect'])
        else:
            button['hover'] = self._is_point_in_rect(self.mouse_x, self.mouse_y, button['rect'])
        
        # Choose button color based on state
        if button['active']:
            color = self.BUTTON_ACTIVE_COLOR
        elif button['hover']:
            color = self.BUTTON_HOVER_COLOR
        else:
            color = self.BUTTON_COLOR
        
        # Draw button background
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 1)  # Border
        
        # Get text size for centering
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        
        # Draw button text
        cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, self.BUTTON_TEXT_COLOR, 1, cv2.LINE_AA)
        
        # Display tooltip on hover if available
        if button['hover'] and button['tooltip']:
            tooltip_y = y + h + 15
            cv2.putText(img, button['tooltip'], (x, tooltip_y), cv2.FONT_HERSHEY_SIMPLEX,
                      0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
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
        
    def _setup_control_panel(self):
        """Create control panel with trackbars"""
        # For exposure, map 0-1000 to 0.1-5000ms logarithmically
        # Convert current exposure to trackbar value
        if self.current_exposure <= 0.1:
            exp_trackbar = 0
        else:
            # Inverse of the mapping function in _exposure_callback
            exp_trackbar = int(1000 * np.log10(self.current_exposure / 0.1) / 3)
            exp_trackbar = max(0, min(1000, exp_trackbar))
            
        cv2.createTrackbar("Exposure", self.control_panel_name, 
                         exp_trackbar, 1000, self._exposure_callback)
        
        # For gain, direct mapping 0-100
        gain_trackbar = int(max(0, min(100, self.current_gain)))
        cv2.createTrackbar("Gain", self.control_panel_name, 
                         gain_trackbar, 100, self._gain_callback)
    
    def _create_control_panel_buttons(self):
        """Create buttons for the control panel"""
        self.control_buttons = []  # Clear existing buttons
        
        # Button dimensions
        button_height = 30
        button_width = self.control_panel_width - 40
        button_spacing = 10
        start_y = 150  # Start below trackbars
        start_x = 20
        
        # Create buttons with actions
        self._create_button("Save Image", start_x, start_y, button_width, button_height, 
                          action=self._save_image, tooltip="Save current image to disk", panel="control")
        
        self._create_button("Toggle Laplacian", start_x, start_y + button_height + button_spacing, 
                          button_width, button_height, action=self._toggle_laplacian, 
                          active=self.show_laplacian, tooltip="Show/hide Laplacian view", panel="control")
        
        self._create_button("Display Mode", start_x, start_y + 2*(button_height + button_spacing), 
                          button_width, button_height, action=self._cycle_display_mode, 
                          tooltip="Change image display mode", panel="control")
        
        self._create_button("Contrast Mode", start_x, start_y + 3*(button_height + button_spacing), 
                          button_width, button_height, action=self._cycle_contrast_mode,
                          tooltip="Change contrast enhancement mode", panel="control")
        
        self._create_button("Grid Mode", start_x, start_y + 4*(button_height + button_spacing), 
                          button_width, button_height, action=self._cycle_grid_mode,
                          tooltip="Change grid overlay mode", panel="control")
        
        self._create_button("RMS Plot", start_x, start_y + 5*(button_height + button_spacing), 
                          button_width, button_height, action=self._toggle_rms_plot,
                          active=self.processor.show_rms_plot, tooltip="Show/hide focus measurement plot", panel="control")
        
        self._create_button("Reset RMS", start_x, start_y + 6*(button_height + button_spacing), 
                          button_width, button_height, action=self._reset_rms_plot,
                          tooltip="Reset RMS plot scaling", panel="control")
        
        self._create_button("Color Mode", start_x, start_y + 7*(button_height + button_spacing), 
                          button_width, button_height, action=self._toggle_color_mode,
                          tooltip="Toggle between color and grayscale", panel="control")
        
        self._create_button("Help", start_x, start_y + 8*(button_height + button_spacing), 
                          button_width, button_height, action=self._toggle_help_menu,
                          tooltip="Show help information", panel="control")
        
        self._create_button("Exit", start_x, start_y + 9*(button_height + button_spacing), 
                          button_width, button_height, action=self._exit_program,
                          tooltip="Exit the application", panel="control")
    
    def _create_main_window_buttons(self, width, height):
        """Create quick access buttons for the main window"""
        self.buttons = []  # Clear existing buttons
        
        # Button dimensions - smaller for main window to be unobtrusive
        button_height = 30
        button_width = 120
        button_spacing = 10
        
        # Position at the top of the window
        start_y = 20
        
        # Create a few essential buttons on main window
        # Space them evenly across the top
        button_count = 5
        total_width = button_count * button_width + (button_count - 1) * button_spacing
        start_x = (width - total_width) // 2
        
        # Add most important buttons to main window
        self._create_button("Save", start_x, start_y, button_width, button_height, 
                          action=self._save_image, tooltip="Save current image")
        
        self._create_button("Laplacian", start_x + button_width + button_spacing, start_y, 
                          button_width, button_height, action=self._toggle_laplacian,
                          active=self.show_laplacian, tooltip="Toggle edge detection view")
        
        self._create_button("Display", start_x + 2*(button_width + button_spacing), start_y, 
                          button_width, button_height, action=self._cycle_display_mode,
                          tooltip="Change display mode")
        
        self._create_button("Help", start_x + 3*(button_width + button_spacing), start_y, 
                          button_width, button_height, action=self._toggle_help_menu,
                          tooltip="Show help menu")
        
        self._create_button("Exit", start_x + 4*(button_width + button_spacing), start_y, 
                          button_width, button_height, action=self._exit_program,
                          tooltip="Exit program")
    
    def _save_image(self):
        """Save current image"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_dir = "./saved"
        os.makedirs(save_dir, exist_ok=True)
        base_filename = os.path.join(save_dir, f"mars_capture_{timestamp}")
        self.processor.save_images(self.last_processed_data, base_filename)
        print(f"Images saved with prefix: {base_filename}")
    
    def _toggle_laplacian(self):
        """Toggle Laplacian view"""
        self.show_laplacian = not self.show_laplacian
        # Update button active state in both windows
        for button_list in [self.buttons, self.control_buttons]:
            for button in button_list:
                if button['label'] == "Toggle Laplacian" or button['label'] == "Laplacian":
                    button['active'] = self.show_laplacian
        print(f"Laplacian view {'enabled' if self.show_laplacian else 'disabled'}")
    
    def _cycle_display_mode(self):
        """Cycle through display modes"""
        mode_name = self.processor.cycle_display_mode()
        print(f"Display mode changed to: {mode_name}")
    
    def _cycle_contrast_mode(self):
        """Cycle through contrast modes"""
        contrast_name = self.processor.cycle_contrast_mode()
        print(f"Contrast mode changed to: {contrast_name}")
    
    def _cycle_grid_mode(self):
        """Cycle through grid modes"""
        grid_name = self.processor.cycle_grid_mode()
        print(f"Grid mode changed to: {grid_name}")
    
    def _toggle_rms_plot(self):
        """Toggle RMS plot"""
        is_visible = self.processor.toggle_rms_plot()
        # Update button active state
        for button in self.control_buttons:
            if button['label'] == "RMS Plot":
                button['active'] = is_visible
                break
        print(f"RMS plot {'enabled' if is_visible else 'disabled'}")
    
    def _reset_rms_plot(self):
        """Reset RMS plot scaling"""
        if self.processor.reset_rms_plot_scale():
            print("RMS plot scaling reset")
        else:
            print("RMS plot not active")
    
    def _toggle_color_mode(self):
        """Toggle color mode"""
        mode_name = self.processor.toggle_color_mode()
        print(f"Display mode changed to: {mode_name}")
    
    def _toggle_help_menu(self):
        """Toggle help menu display"""
        self.show_help_menu = not self.show_help_menu
        print(f"Help menu {'shown' if self.show_help_menu else 'hidden'}")
    
    def _exit_program(self):
        """Exit the application"""
        self.running = False
        print("Exiting application...")
    
    def _show_help_popup(self, display_frame):
        """Draw help popup overlay on the display frame"""
        if not self.show_help_menu:
            return display_frame
        
        h, w = display_frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = display_frame.copy()
        popup_w, popup_h = int(w * 0.8), int(h * 0.8)
        popup_x, popup_y = (w - popup_w) // 2, (h - popup_h) // 2
        
        # Create semi-transparent dark background
        cv2.rectangle(overlay, (popup_x, popup_y), 
                    (popup_x + popup_w, popup_y + popup_h), 
                    (30, 30, 35), -1)
        
        # Add title
        title = "Mars Camera - Help Menu"
        cv2.putText(overlay, title, (popup_x + 20, popup_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 2, cv2.LINE_AA)
        
        # Add help text - keyboard shortcuts
        help_items = [
            ("ESC", "Exit program"),
            ("s", "Save current image"),
            ("+ / -", "Increase/decrease exposure time"),
            ("[ / ]", "Decrease/increase gain"),
            ("d", "Cycle through display modes"),
            ("c", "Cycle through contrast modes"),
            ("g", "Cycle through grid modes"),
            ("r", "Reset RMS plot scaling"),
            ("p", "Toggle real-time RMS plot window"),
            ("l", "Toggle Laplacian filter view"),
            ("m", "Toggle color mode (monochrome/color)")
        ]
        
        start_y = popup_y + 80
        line_height = 30
        col_width = 150
        
        # Draw two columns of help items
        for i, (key, desc) in enumerate(help_items):
            # Determine column and row
            col = i // 6
            row = i % 6
            
            x = popup_x + 30 + col * (col_width + 100)
            y = start_y + row * line_height
            
            # Draw key in highlight color
            cv2.putText(overlay, key, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 1, cv2.LINE_AA)
            
            # Draw description
            cv2.putText(overlay, desc, (x + col_width, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Add footer with close instruction
        footer = "Click anywhere or press 'h' to close this help menu"
        footer_y = popup_y + popup_h - 30
        
        cv2.putText(overlay, footer, (popup_x + 20, footer_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Add a border
        cv2.rectangle(overlay, (popup_x, popup_y), 
                    (popup_x + popup_w, popup_y + popup_h), 
                    (0, 140, 255), 2)
        
        # Blend with original frame (70% help menu, 30% original frame)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, display_frame, 1-alpha, 0, display_frame)
        
        return display_frame
    
    def print_help(self):
        """Print help information to console"""
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
        
        # Set up the control panel
        self._setup_control_panel()
        self._create_control_panel_buttons()
            
        # Start streaming
        if not self.camera.start_streaming():
            print("Failed to start streaming")
            return False
            
        self.print_help()
        self.running = True
        
        # Initialize last processed data (for saving)
        self.last_processed_data = None
        
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
    
    def _update_control_panel(self, camera_info, fps_info):
        """Update the control panel with current information"""
        # Create control panel image
        control_panel = np.zeros((600, self.control_panel_width, 3), dtype=np.uint8)
        control_panel[:] = self.PANEL_COLOR
        
        # Add title
        cv2.putText(control_panel, "Mars Camera", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2, cv2.LINE_AA)
        
        # Add current values
        y_start = 70
        line_height = 25
        
        # Camera info
        cv2.putText(control_panel, f"Exposure: {camera_info['exposure']:.2f} ms", (20, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        
        cv2.putText(control_panel, f"Gain: {camera_info['gain']:.1f}", (20, y_start + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        
        cv2.putText(control_panel, f"Resolution: {camera_info['resolution']}", (20, y_start + 2*line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        
        # Performance info
        perf_y = y_start + 3*line_height
        cv2.putText(control_panel, f"Display FPS: {fps_info['display_fps']:.1f}", (20, perf_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        
        cv2.putText(control_panel, f"Camera FPS: {fps_info['camera_fps']:.1f}", (20, perf_y + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        
        # Add temperature if available
        if 'temperature' in camera_info and camera_info['temperature'] is not None:
            cv2.putText(control_panel, f"Temp: {camera_info['temperature']:.1f}°C", (20, perf_y + 2*line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        
        # Focus info
        focus_y = perf_y + 3*line_height
        cv2.putText(control_panel, f"Focus score: {camera_info['rms']:.2f}", (20, focus_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        
        # Draw RMS plot in control panel
        plot_y = focus_y + 20
        plot_h = 80
        plot_w = self.control_panel_width - 40
        plot_x = 20
        
        # Draw plot background
        cv2.rectangle(control_panel, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (30, 30, 30), -1)
        cv2.rectangle(control_panel, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (70, 70, 70), 1)
        
        # Draw RMS trend if we have values
        if self.processor.rms_values and len(self.processor.rms_values) > 1:
            for i in range(1, len(self.processor.rms_values)):
                y1 = plot_y + plot_h - int((self.processor.rms_values[i-1] / self.processor.max_rms) * (plot_h - 10))
                y2 = plot_y + plot_h - int((self.processor.rms_values[i] / self.processor.max_rms) * (plot_h - 10))
                y1 = max(plot_y, min(plot_y + plot_h, y1))
                y2 = max(plot_y, min(plot_y + plot_h, y2))
                x1 = plot_x + int((i-1) * plot_w / max(1, len(self.processor.rms_values)-1))
                x2 = plot_x + int(i * plot_w / max(1, len(self.processor.rms_values)-1))
                cv2.line(control_panel, (x1, y1), (x2, y2), (0, 140, 255), 1, cv2.LINE_AA)
            
            # Draw horizontal line at current RMS value
            current_y = plot_y + plot_h - int((camera_info['rms'] / self.processor.max_rms) * (plot_h - 10))
            current_y = max(plot_y, min(plot_y + plot_h, current_y))
            cv2.line(control_panel, (plot_x, current_y), (plot_x + plot_w, current_y), 
                    (50, 200, 50), 1, cv2.LINE_AA)
        
        # Draw buttons
        for button in self.control_buttons:
            self._draw_button(control_panel, button, is_control_panel=True)
        
        # Display the control panel
        cv2.imshow(self.control_panel_name, control_panel)
    
    def run(self, save_dir="."):
        """Main processing and display loop"""
        if not self.running:
            return
            
        # Setup FPS calculation
        frame_count = 0
        start_time = time.time()
        
        # Main loop
        self.last_processed_data = None
        
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
                # Handle key presses even when no new frame
                self._handle_key_press(k)
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
            self.last_processed_data = processed_data
            
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
            
            # Add info overlay to frame (with minimal info since we have dedicated control panel)
            # Use modified version with minimal overlay
            display_frame = self._add_minimal_overlay(
                processed_data['display'], camera_info, fps_info)
            
            # Create a combined display with main image and possible Laplacian
            main_display = self._create_main_display(display_frame, processed_data)
            
            # Add GUI buttons on top of the main display
            h, w = main_display.shape[:2]
            
            # Update GUI button positions based on the current display size
            if not self.buttons:
                self._create_main_window_buttons(w, h)
            
            # Draw buttons on the main display
            for button in self.buttons:
                self._draw_button(main_display, button)
                
            # Draw help menu popup if shown
            if self.show_help_menu:
                main_display = self._show_help_popup(main_display)
            
            # Update the separate control panel window
            self._update_control_panel(camera_info, fps_info)
            
            # Display the main window
            cv2.imshow(self.window_name, main_display)
            
            # Check for key press and handle commands
            k = cv2.waitKey(1)
            self._handle_key_press(k)
            
        # Clean up
        self.stop()
    
    def _add_minimal_overlay(self, frame, camera_info, fps_info):
        """Add minimal information overlay to frame"""
        # Add grid overlay if needed
        display_frame = self.processor.add_grid_overlay(frame)
        
        # Add only critical info in corner
        margin = 10
        cv2.putText(display_frame, f"RMS: {camera_info['rms']:.2f}", 
                   (margin, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.putText(display_frame, f"FPS: {fps_info['display_fps']:.1f}", 
                   (margin, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        
        return display_frame
    
    def _handle_key_press(self, k):
        """Handle keyboard input"""
        if k == -1:  # No key pressed
            return
            
        if k == 27:  # ESC key
            self.running = False
        elif k == ord('s'):  # Save image
            self._save_image()
        elif k == ord('h'):  # Help
            self._toggle_help_menu()
        elif k == ord('d'):  # Change display mode
            self._cycle_display_mode()
        elif k == ord('c'):  # Change contrast mode
            self._cycle_contrast_mode()
        elif k == ord('g'):  # Change grid mode
            self._cycle_grid_mode()
        elif k == ord('r'):  # Reset RMS plot scaling
            self._reset_rms_plot()
        elif k == ord('p'):  # Toggle RMS plot
            self._toggle_rms_plot()
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
        elif k == ord('l'):  # Toggle Laplacian view
            self._toggle_laplacian()
        elif k == ord('m'):  # Toggle color mode (monochrome/color)
            self._toggle_color_mode()
    
    def _create_main_display(self, main_frame, processed_data):
        """Create a combined display with main image and Laplacian"""
        # Get dimensions of main frame
        h, w = main_frame.shape[:2]
        
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
            combined_display = np.hstack((main_frame, laplacian_resized))
        else:
            combined_display = main_frame
        
        return combined_display
    
    def stop(self):
        """Stop the viewer and clean up resources"""
        self.running = False
        
        # Stop camera streaming
        if self.camera.streaming:
            self.camera.stop_streaming()
            
        # Close all windows
        cv2.destroyAllWindows()