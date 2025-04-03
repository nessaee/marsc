#!/usr/bin/env python3

import cv2
import numpy as np
import logging
import time
import threading
import matplotlib
# Try to use a backend that works well with OpenCV, but fall back gracefully
try:
    # First try TkAgg as it's widely available
    matplotlib.use('TkAgg')
except ImportError:
    try:
        # Then try Qt5Agg
        matplotlib.use('Qt5Agg')
    except ImportError:
        # Finally fall back to the default backend
        pass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configure logging
logger = logging.getLogger('marsc.image_processor')

# Don't set level here - it will be configured by the main application

class ImageProcessor:
    """Processes camera frames for display and analysis"""
    
    # Display modes
    DISPLAY_AUTO = 0
    DISPLAY_FULL_RANGE = 1
    DISPLAY_NATIVE = 2
    DISPLAY_BIT_SHIFT = 3
    
    # Contrast modes
    CONTRAST_NORMAL = 0
    CONTRAST_ENHANCED = 1
    CONTRAST_HIGH = 2
    
    # Grid modes
    GRID_NONE = 0
    GRID_RULE_OF_THIRDS = 1
    GRID_GOLDEN_RATIO = 2
    GRID_FINE = 3
    
    # Display color modes
    COLOR_MODE_GRAYSCALE = 0
    COLOR_MODE_COLOR = 1
    
    # Raw image modes
    RAW_MODE_BAYER = 0    # Raw image is Bayer pattern (color)
    RAW_MODE_MONOCHROME = 1  # Raw image is monochrome
    
    def __init__(self):
        logger.info("Initializing ImageProcessor")
        self.display_mode = self.DISPLAY_FULL_RANGE
        self.contrast_mode = self.CONTRAST_NORMAL
        self.grid_mode = self.GRID_NONE
        self.color_mode = self.COLOR_MODE_GRAYSCALE  # Default to grayscale as per previous preference
        self.raw_mode = self.RAW_MODE_BAYER  # Default to Bayer pattern (color sensor)
        self.rms_values = []
        self.max_rms = 20
        self.show_rms_plot = False  # Flag to control RMS plot visibility
        self.rms_plot_initialized = False  # Track if the plot has been initialized
        self.rms_fig = None  # Figure for the RMS plot
        self.rms_ax = None   # Axis for the RMS plot
        self.rms_line = None  # Line object for the RMS plot
        self.rms_animation = None  # Animation object for real-time updates
        self.rms_x_data = []  # X-axis data (frame numbers)
        self.frame_count = 0  # Counter for frames processed
        
        logger.debug(f"Initial settings: display_mode={self.get_display_mode_name()}, "
                    f"contrast_mode={self.get_contrast_mode_name()}, "
                    f"grid_mode={self.get_grid_mode_name()}, "
                    f"color_mode={self.get_color_mode_name()}, "
                    f"raw_mode={self.get_raw_mode_name()}")
        
    def get_display_mode_name(self):
        """Get the name of the current display mode"""
        mode_names = ["Auto", "Full dynamic range", "Native bit depth", "Bit shift"]
        return mode_names[self.display_mode]
    
    def get_contrast_mode_name(self):
        """Get the name of the current contrast mode"""
        contrast_modes = ["Normal contrast", "Enhanced contrast", "High contrast"]
        return contrast_modes[self.contrast_mode]
    
    def get_grid_mode_name(self):
        """Get the name of the current grid mode"""
        grid_modes = ["No Grid", "Rule of Thirds", "Golden Ratio", "Fine Grid"]
        return grid_modes[self.grid_mode]
        
    def get_color_mode_name(self):
        """Get the name of the current color mode"""
        color_modes = ["Grayscale", "Color"]
        return color_modes[self.color_mode]
        
    def get_raw_mode_name(self):
        """Get the name of the current raw mode"""
        raw_modes = ["Bayer Pattern", "Monochrome"]
        return raw_modes[self.raw_mode]
        
    def initialize_rms_plot(self):
        """Initialize the RMS plot window"""
        if self.rms_plot_initialized:
            return
            
        logger.info("Initializing RMS plot window")
        
        # Start the plot in a separate thread to avoid blocking the main UI
        self.plot_thread = threading.Thread(target=self._create_plot_window)
        self.plot_thread.daemon = True  # Thread will exit when main program exits
        self.plot_thread.start()
        
        # Mark as initialized
        self.rms_plot_initialized = True
        
    def _create_plot_window(self):
        """Create the plot window in a separate thread"""
        try:
            # Create a new figure with a unique number to ensure it's a separate window
            self.rms_fig = plt.figure(figsize=(8, 4), num="RMS Plot")
            self.rms_ax = self.rms_fig.add_subplot(111)
            
            # Set window title if the backend supports it
            try:
                self.rms_fig.canvas.manager.set_window_title('Focus Measurement (RMS) Plot')
            except (AttributeError, NotImplementedError):
                # Some backends don't support setting window title
                pass
            
            # Initialize with empty data
            self.rms_line, = self.rms_ax.plot([], [], 'b-', linewidth=2)
            
            # Configure the plot
            self.rms_ax.set_title('Real-time Focus Measurement')
            self.rms_ax.set_xlabel('Frame')
            self.rms_ax.set_ylabel('RMS Value')
            self.rms_ax.grid(True)
            
            # Set initial y-axis limits
            self.rms_ax.set_ylim(0, max(20, self.max_rms * 1.2))
            
            # Create animation that updates every 100ms
            # Set save_count to 100 to limit the number of cached frames
            self.rms_animation = FuncAnimation(
                self.rms_fig, self._update_rms_plot, interval=100, blit=True, save_count=100)
            
            # Show the plot window - this will block in this thread
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Failed to create RMS plot window: {e}")
            self.rms_plot_initialized = False
        
    def _update_rms_plot(self, frame):
        """Update the RMS plot with new data"""
        try:
            if not self.rms_values or not hasattr(self, 'rms_line') or self.rms_line is None:
                return [self.rms_line] if hasattr(self, 'rms_line') and self.rms_line is not None else []
                
            # Update the line data
            self.rms_line.set_data(self.rms_x_data, self.rms_values)
            
            # Adjust x-axis limits to show all data
            x_min = max(0, self.frame_count - 100)
            self.rms_ax.set_xlim(x_min, self.frame_count + 5)
            
            # Adjust y-axis if needed
            current_max = max(self.rms_values) if self.rms_values else 20
            if current_max > self.rms_ax.get_ylim()[1] * 0.8:
                self.rms_ax.set_ylim(0, current_max * 1.2)
            
            return [self.rms_line]
        except Exception as e:
            logger.error(f"Error updating RMS plot: {e}")
            return []
        
    def toggle_rms_plot(self):
        """Toggle the RMS plot window on/off"""
        try:
            self.show_rms_plot = not self.show_rms_plot
            
            if self.show_rms_plot:
                # Initialize the plot if needed
                if not self.rms_plot_initialized:
                    self.initialize_rms_plot()
                    logger.info("RMS plot window initialized")
                # The plot is shown automatically when initialized
            else:
                logger.info("RMS plot updates paused")
                # We can't hide the plot once it's shown due to threading,
                # but we can stop updating it
                    
            return self.show_rms_plot
        except Exception as e:
            logger.error(f"Error toggling RMS plot: {e}")
            return False
        
    def reset_rms_plot_scale(self):
        """Reset the RMS plot scale"""
        try:
            if self.rms_plot_initialized and hasattr(self, 'rms_ax') and self.rms_ax is not None:
                self.max_rms = max(20, max(self.rms_values) if self.rms_values else 20)
                self.rms_ax.set_ylim(0, self.max_rms * 1.2)
                logger.info(f"RMS plot scale reset to max value: {self.max_rms:.2f}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error resetting RMS plot scale: {e}")
            return False
        
    def set_raw_mode(self, is_monochrome):
        """Set the raw mode based on whether the sensor is monochrome"""
        mode_name = "Monochrome" if is_monochrome else "Bayer Pattern"
        logger.info(f"Setting raw mode to: {mode_name}")
        self.raw_mode = self.RAW_MODE_MONOCHROME if is_monochrome else self.RAW_MODE_BAYER
        return self.get_raw_mode_name()
        
    def cycle_display_mode(self):
        """Cycle through display modes"""
        self.display_mode = (self.display_mode + 1) % 4
        return self.get_display_mode_name()
    
    def cycle_contrast_mode(self):
        """Cycle through contrast modes"""
        self.contrast_mode = (self.contrast_mode + 1) % 3
        return self.get_contrast_mode_name()
    
    def cycle_grid_mode(self):
        """Cycle through grid modes"""
        self.grid_mode = (self.grid_mode + 1) % 4
        return self.get_grid_mode_name()
    
    def toggle_color_mode(self):
        """Toggle between grayscale and color display modes"""
        self.color_mode = 1 - self.color_mode  # Toggle between 0 and 1
        return self.get_color_mode_name()
    
    def reset_rms_plot(self):
        """Reset RMS plot scaling"""
        self.max_rms = 20
        self.rms_values = []
        
    def convert_12bit_bayer_to_8bit_gray(self, raw_image):
        """Convert 12-bit Bayer pattern to 8-bit grayscale with antialiasing
        
        Args:
            raw_image: 12-bit raw Bayer pattern image (BGGR format)
            
        Returns:
            8-bit grayscale image with preserved detail and reduced aliasing
        """
        # Ensure input is in the right format
        if raw_image.dtype != np.uint16 and raw_image.max() > 255:
            # Convert to uint16 if not already and preserve bit depth
            raw_image = raw_image.astype(np.uint16)
        
        # Apply anti-aliasing filter to reduce checkerboard artifacts
        # This preprocessing helps reduce Moiré and aliasing in the demosaicing stage
        if raw_image.dtype == np.uint16:
            # For 12-bit data in uint16 format
            
            # Apply a mild Gaussian blur to reduce aliasing (this avoids checkerboard patterns)
            # The blur is applied to the raw Bayer data before demosaicing for better results
            # We use a small kernel to avoid excessive softening
            raw_image_prefiltered = cv2.GaussianBlur(raw_image, (3, 3), 0.5)
            
            # Convert to 8-bit with proper scaling
            raw_image_8bit = cv2.convertScaleAbs(raw_image_prefiltered, alpha=255.0/4095)
            
            # Alternatively apply a direct median blur for aliasing prevention if needed
            # Uncomment if checkerboard artifacts persist with Gaussian
            # raw_image_8bit = cv2.medianBlur(raw_image_8bit, 3)
        else:
            # For already 8-bit data
            # Still apply anti-aliasing to raw Bayer pattern
            raw_image_8bit = cv2.GaussianBlur(raw_image, (3, 3), 0.5)
            
        # Debayer the image using a high-quality algorithm
        try:
            # Use enhanced interpolation to prevent checkerboard artifacts
            # First attempt with EA (Edge-Aware) algorithm which is better at preventing checkerboard patterns
            # Fall back to VNG which is still effective at reducing aliasing
            try:
                # EA provides better interpolation especially around edges
                rgb_image = cv2.cvtColor(raw_image_8bit, cv2.COLOR_BAYER_RGGB2RGB_EA)
            except Exception:
                # Fall back to VNG if EA is not available (older OpenCV versions)
                rgb_image = cv2.cvtColor(raw_image_8bit, cv2.COLOR_BAYER_RGGB2RGB_VNG)
            
            # Convert to grayscale using proper luminance weights (matches human perception)
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            
            return gray_image
            
        except Exception as e:
            print(f"Advanced debayering failed: {e}, falling back to simpler method")
            try:
                # Fall back to basic debayering
                rgb_image = cv2.cvtColor(raw_image_8bit, cv2.COLOR_BAYER_RGGB2RGB)
                gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
                return gray_image
            except Exception as e2:
                print(f"Simple debayering also failed: {e2}, using direct conversion")
                # If all debayering fails, return a simple scaled version of the raw image
                # Use direct downsampling to avoid introducing processing artifacts
                if raw_image.dtype != np.uint8:
                    # Linear scaling from 12-bit to 8-bit
                    max_val = float((1 << 12) - 1)  # 12-bit max (4095)
                    scaled = (raw_image.astype(np.float32) / max_val) * 255.0
                    return np.clip(scaled, 0, 255).astype(np.uint8)
                return raw_image_8bit
    
    def convert_12bit_bayer_to_color(self, raw_image):
        """Convert 12-bit Bayer to 8-bit color image without any blur
        
        Args:
            raw_image: 12-bit raw Bayer pattern image (RGGB format)
            
        Returns:
            8-bit color RGB image
        """
        try:
            # Log input image properties
            logger.debug(f"Converting raw image: shape={raw_image.shape}, dtype={raw_image.dtype}, "
                        f"min={np.min(raw_image)}, max={np.max(raw_image)}, "
                        f"mode={self.get_raw_mode_name()}")
            
            # Check if we should treat this as monochrome
            if self.raw_mode == self.RAW_MODE_MONOCHROME:
                logger.debug("Processing as monochrome image")
                # For monochrome sensor, just convert to 8-bit and create RGB image
                if raw_image.dtype != np.uint8:
                    # Scale from 12-bit to 8-bit
                    logger.debug("Converting from high bit depth to 8-bit")
                    # Direct linear scaling from 12-bit to 8-bit with exact division
                    gray_8bit = np.clip((raw_image.astype(np.float32) / 4095.0 * 255.0), 0, 255).astype(np.uint8)
                else:
                    gray_8bit = raw_image.copy()
                    
                # Create RGB image from grayscale
                logger.debug("Converting grayscale to RGB for display")
                return cv2.cvtColor(gray_8bit, cv2.COLOR_GRAY2RGB)
            
            # For Bayer pattern (color sensor)
            logger.debug("Processing as Bayer pattern image")
            
            # Ensure input is in the right format
            if raw_image.dtype != np.uint16 and raw_image.max() > 255:
                # Convert to uint16 if not already and preserve bit depth
                logger.debug(f"Converting to uint16 from {raw_image.dtype}")
                raw_image = raw_image.astype(np.uint16)
            
            # No blur preprocessing - direct conversion
            if raw_image.dtype == np.uint16:
                # For 12-bit data in uint16 format, convert to 8-bit with proper scaling
                logger.debug("Scaling from 12-bit to 8-bit")
                # Direct linear scaling from 12-bit to 8-bit with exact division
                raw_image_8bit = np.clip((raw_image.astype(np.float32) / 4095.0 * 255.0), 0, 255).astype(np.uint8)
            else:
                # For already 8-bit data, use as is
                logger.debug("Using existing 8-bit data")
                raw_image_8bit = raw_image.copy()
                
            # Direct debayering without any additional filtering
            try:
                # Use standard debayering algorithm
                logger.debug("Applying standard RGGB debayering")
                start_time = time.time()
                rgb_image = cv2.cvtColor(raw_image_8bit, cv2.COLOR_BAYER_RGGB2RGB)
                logger.debug(f"Debayering completed in {(time.time() - start_time)*1000:.1f}ms")
                
                # Log output image properties
                logger.debug(f"Debayered image: shape={rgb_image.shape}, dtype={rgb_image.dtype}, "
                            f"min={np.min(rgb_image)}, max={np.max(rgb_image)}")
                return rgb_image
            except Exception as e:
                # Fall back to standard algorithm if EA/VNG is not available
                logger.warning(f"First debayering attempt failed: {e}, trying alternative method")
                rgb_image = cv2.cvtColor(raw_image_8bit, cv2.COLOR_BAYER_RGGB2RGB)
                return rgb_image
            
        except Exception as e:
            logger.error(f"Color conversion failed: {e}, falling back to simpler method")
            try:
                # Fall back to basic debayering
                logger.debug("Using fallback conversion method")
                if raw_image.dtype != np.uint8:
                    # Scale from 12-bit to 8-bit
                    logger.debug("Scaling high bit-depth to 8-bit in fallback path")
                    # Direct linear scaling from 12-bit to 8-bit with exact division
                    raw_image_8bit = np.clip((raw_image.astype(np.float32) / 4095.0 * 255.0), 0, 255).astype(np.uint8)
                else:
                    raw_image_8bit = raw_image.copy()
                
                # For monochrome mode, just convert to RGB
                if self.raw_mode == self.RAW_MODE_MONOCHROME:
                    logger.debug("Fallback: Converting monochrome to RGB")
                    return cv2.cvtColor(raw_image_8bit, cv2.COLOR_GRAY2RGB)
                    
                # For Bayer pattern, use standard debayering
                logger.debug("Fallback: Applying standard debayering")
                rgb_image = cv2.cvtColor(raw_image_8bit, cv2.COLOR_BAYER_RGGB2RGB)
                return rgb_image
            except Exception as e2:
                logger.error(f"Simple color conversion also failed: {e2}, using last resort method")
                # Last resort: create a grayscale image and convert to RGB
                try:
                    logger.debug("Last resort: Treating as grayscale and converting to RGB")
                    return cv2.cvtColor(raw_image_8bit, cv2.COLOR_GRAY2RGB)
                except Exception as e3:
                    logger.critical(f"All conversion methods failed: {e3}")
                    return None
        
    def process_frame(self, frame, bit_depth):
        """Process a camera frame for display"""
        start_time = time.time()
        logger.debug(f"Processing frame: bit_depth={bit_depth}")
        
        if frame is None:
            logger.warning("Received None frame")
            return None
            
        # Keep original for processing and saving
        original = frame.copy()
        logger.debug(f"Frame properties: shape={frame.shape}, dtype={frame.dtype}, "
                    f"min={np.min(frame)}, max={np.max(frame)}")
        
        # Handle RGGB Bayer pattern if the image is raw
        # Sony IMX462 uses RGGB Bayer matrix pattern
        if bit_depth == 12 and len(frame.shape) == 2:  # Raw image from 12-bit sensor
            try:
                # First convert to 8-bit for processing
                # Ensure we use the correct max value for 12-bit data (4095)
                max_val = 4095.0  # 12-bit max value (2^12 - 1)
                # Use simple linear scaling with proper floating point division
                frame_8bit = np.clip((frame.astype(np.float32) / max_val * 255.0), 0, 255).astype(np.uint8)
                
                # Check if we're treating this as monochrome or Bayer pattern
                logger.debug(f"Processing raw image with mode: {self.get_raw_mode_name()}")
                if self.raw_mode == self.RAW_MODE_MONOCHROME:
                    # For monochrome sensor, just use the raw data directly
                    gray = frame_8bit.copy()
                    scaling_method = "Direct Monochrome"
                    
                    # Create a color version (still grayscale) for consistent processing
                    color_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                else:
                    # For Bayer pattern, apply 2x2 binning to properly handle the RGGB pattern
                    # This eliminates the grid pattern by properly combining the RGGB quad
                    h, w = frame_8bit.shape
                    # Ensure even dimensions
                    h_even = h - (h % 2)
                    w_even = w - (w % 2)
                    
                    # Reshape to separate the 2x2 Bayer blocks
                    reshaped = frame_8bit[:h_even, :w_even].reshape(h_even//2, 2, w_even//2, 2)
                    
                    # Average the 2x2 blocks to create a properly debayered grayscale image
                    # This directly combines R+G+G+B pixels in each 2x2 block
                    gray = reshaped.mean(axis=(1, 3)).astype(np.uint8)
                    
                    # Set scaling method for this path
                    scaling_method = "2x2 Bayer Binning"
                    
                    # For color display, use OpenCV's debayering without any blur
                    try:
                        # No blur - use raw data directly
                        # Use the most robust algorithm available - convert directly to RGB instead of BGR
                        color_frame = cv2.cvtColor(frame_8bit, cv2.COLOR_BAYER_RGGB2RGB)
                        
                        # Store both the original debayered frame and the display frame
                        self.original_debayered = color_frame.copy()
                        self.debayered_frame = color_frame
                    except Exception as e:
                        print(f"Color debayering failed: {e}")
                        # Create a color version from our grayscale - use RGB format
                        color_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                        self.original_debayered = color_frame.copy()
                        self.debayered_frame = color_frame
            except Exception as e:
                logger.error(f"Enhanced debayering failed: {e}, falling back to simple conversion")
                # Fall back to simple scaling if enhanced debayering fails
                # Use exact value for 12-bit max to avoid potential bit-shift errors
                max_val = 4095.0  # 12-bit max value (2^12 - 1)
                scale_factor = 255.0 / max_val
                logger.debug("Using simple scaling for 12-bit to 8-bit conversion")
                # Ensure proper floating point calculation and clipping
                gray = np.clip(frame.astype(np.float32) * scale_factor, 0, 255).astype(np.uint8)
                color_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                self.original_debayered = color_frame.copy()
                self.debayered_frame = color_frame
                
                # Set scaling method for fallback path
                scaling_method = "Simple 12-bit Scaling"
        
        # Get value range info
        frame_min = np.min(frame)
        frame_max = np.max(frame)
        dynamic_range = frame_max - frame_min
        max_theoretical_val = float((1 << min(bit_depth, 24)) - 1)
        percent_of_max = min(100, int(100.0 * dynamic_range / max_theoretical_val))
        
        # Initialize scaling_method to ensure it's always defined
        scaling_method = "Direct Debayer"
        
        # Skip this step if we've already created a properly debayered grayscale image
        if 'gray' not in locals() or gray is None:
            # Use full dynamic range for best display quality
            if dynamic_range > 0:
                # Scale to 0-255 range (8-bit)
                gray = np.clip(((frame - frame_min) * 255.0 / dynamic_range), 0, 255).astype(np.uint8)
                scaling_method = "8-bit Dynamic Range"
            else:
                # If no dynamic range, just convert directly to 8-bit
                # Use exact value for 12-bit max to avoid potential bit-shift errors
                max_val = 4095.0  # 12-bit max value (2^12 - 1)
                scale_factor = 255.0 / max_val
                # Ensure proper floating point calculation and clipping
                gray = np.clip(frame.astype(np.float32) * scale_factor, 0, 255).astype(np.uint8)
                scaling_method = "8-bit Direct"
            
            # No blur filtering - use raw data directly
            # If this is a fallback path for a Bayer sensor, we'll still need some processing
            if bit_depth == 12 and len(frame.shape) == 2 and self.raw_mode == self.RAW_MODE_BAYER:
                # Simple 2x2 binning for Bayer pattern if possible
                h, w = gray.shape
                if h % 2 == 0 and w % 2 == 0:
                    # Reshape and average to handle Bayer pattern
                    reshaped = gray.reshape(h//2, 2, w//2, 2)
                    gray = reshaped.mean(axis=(1, 3)).astype(np.uint8)
            
        # Always use 8-bit depth for consistency
        
        # Store the original unmodified grayscale image for visualization
        gray_original = gray.copy()
        logger.debug(f"Grayscale image: shape={gray.shape}, min={np.min(gray)}, max={np.max(gray)}")
        
        # Apply contrast enhancement if enabled - but only for display purposes, not for the original image
        if self.contrast_mode == self.CONTRAST_ENHANCED:  # Enhanced contrast
            p_low, p_high = np.percentile(gray, [5, 95])
            if p_high > p_low:
                gray = np.clip(255.0 * (gray - p_low) / (p_high - p_low), 0, 255).astype(np.uint8)
            contrast_str = "Enhanced contrast"
        elif self.contrast_mode == self.CONTRAST_HIGH:  # High contrast (histogram equalization)
            gray = cv2.equalizeHist(gray)
            contrast_str = "High contrast"
        else:  # Normal contrast
            contrast_str = "Normal contrast"

        # Apply simple Laplacian filter for focus measurement
        # Standard 3x3 Laplacian kernel for edge detection
        kernel_size = 3
        kernel = np.array([
            [-1,-1, -1, -1,-1],
            [-1,-1, -1, -1, -1],
            [-1, -1, 24, -1,-1],
            [-1,-1, -1, -1,-1],
            [-1,-1, -1, -1,-1]
        ], dtype=np.float64)
        
        # Use the appropriate grayscale image for Laplacian filter
        if self.raw_mode == self.RAW_MODE_MONOCHROME:
            # For monochrome sensor, use the gray image directly
            laplacian_input = gray
        elif hasattr(self, 'original_debayered') and self.original_debayered is not None:
            # For Bayer pattern, use the debayered grayscale image
            laplacian_input = cv2.cvtColor(self.original_debayered, cv2.COLOR_RGB2GRAY)
        else:
            # Fall back to the gray image if no debayered image is available
            laplacian_input = gray
        
        # Apply the Laplacian to the properly debayered grayscale image
        logger.debug(f"Applying Laplacian filter to {laplacian_input.shape} image")
        laplacian = cv2.filter2D(laplacian_input, cv2.CV_64F, kernel)
        
        # Store the unmodified Laplacian result for both visualization and measurement
        laplacian_raw = laplacian.copy()
        
        # Simple RMS calculation without thresholding
        rms = np.sqrt(np.mean(np.square(laplacian_raw)))
        self.rms_values.append(rms)
        logger.debug(f"Focus measurement: RMS={rms:.3f}")
        
        # Update frame counter for RMS plot
        self.frame_count += 1
        self.rms_x_data.append(self.frame_count)
        
        # Simple max value adjustment
        if rms > self.max_rms:
            self.max_rms = rms * 1.2
        
        # Keep only the last 100 values for display
        if len(self.rms_values) > 100:
            self.rms_values.pop(0)
            self.rms_x_data.pop(0)
            
        # Update RMS plot if it's visible
        if self.show_rms_plot and self.rms_plot_initialized:
            # The actual update happens in the animation loop
            pass
        
        # Choose between color and grayscale display based on user preference
        logger.debug(f"Preparing display frame with color mode: {self.get_color_mode_name()}")
        if self.color_mode == self.COLOR_MODE_COLOR and hasattr(self, 'debayered_frame') and self.debayered_frame is not None:
            # Use color display if color mode is selected and we have a debayered color frame
            display_frame = self.debayered_frame.copy()
        else:
            # Use grayscale display (default or if color data isn't available)
            # Use the original unmodified grayscale image for display
            # No verification or recovery needed - use the original image directly
            
            # Convert grayscale to RGB format for consistent display
            display_frame = cv2.cvtColor(gray_original, cv2.COLOR_GRAY2RGB)
        
        # Simple visualization of Laplacian for focus feedback - no enhancements
        # Take absolute value of Laplacian to get edge magnitude
        laplacian_abs = np.abs(laplacian)
        
        # Simple normalization without gamma correction or additional filtering
        # Just scale to 0-255 range without changing distribution
        max_val = np.max(laplacian_abs)
        if max_val > 0:
            laplacian_enhanced = (laplacian_abs * (255.0 / max_val)).astype(np.uint8)
        else:
            laplacian_enhanced = np.zeros_like(laplacian_abs, dtype=np.uint8)
        
        # Use grayscale visualization for the Laplacian
        # Convert to RGB format (still monochrome) for consistent display
        laplacian_colored = cv2.cvtColor(laplacian_enhanced, cv2.COLOR_GRAY2RGB)
        
        # Do not resize - maintain the same resolution as input camera
        
        # Add a label to indicate this is the focus visualization image
        cv2.putText(laplacian_colored, "Laplacian", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Add meta information to return
        result = {
            'original': original,
            'original_debayered': self.original_debayered if hasattr(self, 'original_debayered') else None,
            'gray': gray_original,  # Use the unmodified grayscale image
            'display': display_frame,
            'laplacian': laplacian,  # Raw Laplacian for calculations
            'laplacian_vis': laplacian_colored,  # Visualized Laplacian
            'rms': rms,
            'frame_min': frame_min,
            'frame_max': frame_max,
            'percent_of_max': percent_of_max,
            'scaling_method': scaling_method,
            'contrast_method': contrast_str
        }
        
        processing_time = (time.time() - start_time) * 1000
        logger.debug(f"Frame processing completed in {processing_time:.1f}ms")
        return result
    
    def add_grid_overlay(self, frame):
        """Add grid overlay to the frame"""
        if self.grid_mode == self.GRID_NONE:
            return frame
            
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        if self.grid_mode == self.GRID_RULE_OF_THIRDS:  # Rule of Thirds
            # Vertical lines
            x1, x2 = int(w/3), int(2*w/3)
            cv2.line(overlay, (x1, 0), (x1, h), (0, 140, 255), 1)
            cv2.line(overlay, (x2, 0), (x2, h), (0, 140, 255), 1)
            # Horizontal lines
            y1, y2 = int(h/3), int(2*h/3)
            cv2.line(overlay, (0, y1), (w, y1), (0, 140, 255), 1)
            cv2.line(overlay, (0, y2), (w, y2), (0, 140, 255), 1)
        elif self.grid_mode == self.GRID_GOLDEN_RATIO:  # Golden Ratio (Phi ≈ 1.618)
            # Vertical lines
            phi = 1.618
            x1 = int(w / (1 + phi))
            x2 = int(w - x1)
            cv2.line(overlay, (x1, 0), (x1, h), (0, 140, 255), 1)
            cv2.line(overlay, (x2, 0), (x2, h), (0, 140, 255), 1)
            # Horizontal lines
            y1 = int(h / (1 + phi))
            y2 = int(h - y1)
            cv2.line(overlay, (0, y1), (w, y1), (0, 140, 255), 1)
            cv2.line(overlay, (0, y2), (w, y2), (0, 140, 255), 1)
        elif self.grid_mode == self.GRID_FINE:  # Fine Grid
            # Draw a 5x5 grid
            for i in range(1, 5):
                # Vertical lines
                x = int(i * w / 5)
                cv2.line(overlay, (x, 0), (x, h), (0, 140, 255), 1)
                # Horizontal lines
                y = int(i * h / 5)
                cv2.line(overlay, (0, y), (w, y), (0, 140, 255), 1)
        
        return overlay
    
    def add_info_overlay(self, frame, camera_info, fps_info):
        """Add information overlay to frame"""
        display_frame = self.add_grid_overlay(frame)
        h, w = display_frame.shape[:2]
        
        # Add FPS and camera info
        cv2.putText(display_frame, f"Display FPS: {fps_info['display_fps']:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Camera FPS: {fps_info['camera_fps']:.1f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"RMS: {camera_info['rms']:.2f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Exp: {camera_info['exposure']:.2f}ms", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Gain: {camera_info['gain']:.1f}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Res: {camera_info['resolution']}", 
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                   
        # Add processing info
        cv2.putText(display_frame, 
                   f"{camera_info['scaling_method']}, {camera_info['contrast_method']}, {self.get_grid_mode_name()}", 
                   (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, 
                   f"Range: {camera_info['frame_min']}-{camera_info['frame_max']} ({camera_info['percent_of_max']}%)", 
                   (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add temperature if available
        if 'temperature' in camera_info and camera_info['temperature'] is not None:
            cv2.putText(display_frame, f"Temp: {camera_info['temperature']:.1f}°C", 
                       (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw RMS plot
        plot_h, plot_w = 100, w
        plot_y = h - plot_h - 10
        
        if plot_y > 0 and self.rms_values:
            # Draw plot background
            cv2.rectangle(display_frame, (0, plot_y), (plot_w, plot_y + plot_h), (20, 20, 20), -1)
            
            # Draw RMS trend if we have values
            if len(self.rms_values) > 1:
                for i in range(1, len(self.rms_values)):
                    y1 = plot_y + plot_h - int((self.rms_values[i-1] / self.max_rms) * (plot_h - 10))
                    y2 = plot_y + plot_h - int((self.rms_values[i] / self.max_rms) * (plot_h - 10))
                    y1 = max(plot_y, min(plot_y + plot_h, y1))
                    y2 = max(plot_y, min(plot_y + plot_h, y2))
                    x1 = int((i-1) * plot_w / max(1, len(self.rms_values)-1))
                    x2 = int(i * plot_w / max(1, len(self.rms_values)-1))
                    cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                # Draw horizontal line at current RMS value
                current_y = plot_y + plot_h - int((camera_info['rms'] / self.max_rms) * (plot_h - 10))
                current_y = max(plot_y, min(plot_y + plot_h, current_y))
                cv2.line(display_frame, (0, current_y), (plot_w, current_y), (0, 128, 255), 1)
            
            # Add scale to RMS plot
            cv2.putText(display_frame, f"Focus Trend (Max: {self.max_rms:.1f})", 
                       (10, plot_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame
    
    def save_images(self, processed_data, base_filename):
        """Save processed images to disk with enhanced quality preservation"""
        import os
        
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(base_filename)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Get bit depth and images from the processed data
        original = processed_data['original']
        gray = processed_data['gray']
        display = processed_data['display']
        laplacian_vis = processed_data['laplacian_vis'] if 'laplacian_vis' in processed_data else None
        
        # Check original image bit depth
        if original.dtype == np.uint16 or original.max() > 255:
            # High bit-depth image (likely 12-bit)
            
            # For 12-bit data, save as 16-bit PNG for full fidelity
            # First convert to proper uint16 format with correct scaling
            if original.dtype != np.uint16:
                original_16bit = original.astype(np.uint16)
            else:
                original_16bit = original.copy()
            
            # If the 12-bit data is stored in 16-bit container, scale it properly to use full 16-bit range
            if np.max(original_16bit) < 4096:  # It's 12-bit data in 16-bit container
                original_16bit = np.clip(original_16bit.astype(np.float32) * (65535/4095), 0, 65535).astype(np.uint16)
            
            # Save full bit-depth image as 16-bit PNG
            cv2.imwrite(f"{base_filename}_full.png", original_16bit, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print(f"Full bit depth image saved as {base_filename}_full.png")
            
            # Create and save a high-quality debayered version with antialiasing
            # This uses our enhanced debayering algorithm
            if len(original.shape) == 2:  # It's a Bayer pattern
                # Debayer to color with high quality
                if hasattr(self, 'original_debayered') and self.original_debayered is not None:
                    color_debayered = self.original_debayered.copy()
                else:
                    color_debayered = self.convert_12bit_bayer_to_color(original)
                
                if color_debayered is not None:
                    # No post-processing
                    
                    # Save high-quality debayered version
                    cv2.imwrite(f"{base_filename}_debayered.png", color_debayered, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    print(f"High-quality debayered image saved as {base_filename}_debayered.png")
        
        # Save standard 8-bit grayscale image without post-processing
        cv2.imwrite(f"{base_filename}_8bit.png", gray, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"8-bit image saved as {base_filename}_8bit.png")
        
        # Save stretched version for better visibility
        stretch_img = cv2.normalize(gray, None, alpha=0, beta=255, 
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        cv2.imwrite(f"{base_filename}_stretched.png", stretch_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"Stretched image saved as {base_filename}_stretched.png")
        
        # Save display view with high quality
        cv2.imwrite(f"{base_filename}_display.png", display, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"Display view saved as {base_filename}_display.png")
        
        # Save focus visualization as grayscale if available
        if laplacian_vis is not None:
            # Convert back to grayscale for saving
            if len(laplacian_vis.shape) == 3:
                laplacian_gray = cv2.cvtColor(laplacian_vis, cv2.COLOR_BGR2GRAY)
            else:
                laplacian_gray = laplacian_vis
                
            cv2.imwrite(f"{base_filename}_laplacian.png", laplacian_gray, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print(f"Laplacian saved as {base_filename}_laplacian.png")
