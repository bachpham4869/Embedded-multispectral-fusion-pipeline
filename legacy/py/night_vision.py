#!/usr/bin/env python3
"""
HYBRID NIGHT VISION - Raspberry Pi 4 + IMX290
Combines Dark Channel Prior with Fast Atmosphere Light
Optimized for real-time (25-35 FPS)
Output: flip horizontal so left/right matches physical (no mirror).
"""

import cv2
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter
from picamera2 import Picamera2
import time
from collections import deque
import os

class HybridNightVision:
    """
    Hybrid Mode: Dark/Bright Channel + Fast Atmosphere Light + Adaptive CLAHE
    No complex transmission maps or dehaze formula
    """
    
    def __init__(self, width=640, height=480, patch_size=5, update_rate=8):
        """
        Initialize hybrid enhancer
        
        Args:
            width: Display width
            height: Display height
            patch_size: Size for dark/bright channel (odd number)
            update_rate: Update atmosphere light every N frames (lower = less ghosting)
        """
        # Processing resolution (small for speed)
        self.proc_width = 320
        self.proc_height = 240
        self.display_width = width
        self.display_height = height
        self.w = patch_size if patch_size % 2 == 1 else patch_size + 1
        self.update_rate = update_rate  # Increased from 15 to 8 for less ghosting
        
        # Parameters
        self.p = 0.05  # Top percentage for atmosphere light (reduced for speed)
        
        # State variables
        self.frame_count = 0
        self.last_A = np.array([0.7, 0.7, 0.7], dtype=np.float32)  # Brighter default
        self.last_dark_map = None
        self.last_bright_map = None
        self.last_weight_map = None
        
        # Buffers for temporal stability
        self.A_buffer = deque(maxlen=10)
        self.brightness_buffer = deque(maxlen=15)
        
        # Adaptive CLAHE objects (pre-created for speed)
        self.clahe_levels = {
            'very_dark': cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)),
            'dark': cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)),
            'medium': cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        }
        
        # Enhancement parameters
        self.detail_strength = 0.25
        self.min_brightness_boost = 1.1
        self.max_brightness_boost = 1.8
        
        # Performance tracking
        self.fps = 0
        self.processing_times = deque(maxlen=100)
        self.frame_counter = 0
        self.last_fps_time = time.time()
        
        # Image saving
        self.save_counter = 0
        self.save_dir = "comparison_images"
        self.create_save_directory()
        
        # Initialize with default values
        self.A_buffer.append(self.last_A)
    
    def create_save_directory(self):
        """Create directory for saving comparison images"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created directory: {self.save_dir}")
    
    def compute_illumination_channels(self, I_float):
        """
        Compute dark and bright channels using optimized filters
        """
        # Get min and max across color channels
        min_channels = np.min(I_float, axis=2)
        max_channels = np.max(I_float, axis=2)
        
        # Use minimum/maximum filters (optimized C implementation)
        dark_map = minimum_filter(min_channels, size=(self.w, self.w), mode='reflect')
        bright_map = maximum_filter(max_channels, size=(self.w, self.w), mode='reflect')
        
        return dark_map, bright_map
    
    def fast_atmosphere_light(self, I_float, bright_map):
        """
        Fast atmosphere light estimation using quantiles (no full sorting)
        """
        # Use quantile to find threshold for top p% (faster than sorting all)
        threshold = np.quantile(bright_map, 1 - self.p)
        
        # Create mask for bright pixels
        mask = bright_map >= threshold
        
        # Calculate mean of bright pixels
        if np.any(mask):
            # Get pixel values where mask is True
            bright_pixels = I_float[mask]
            A = np.mean(bright_pixels, axis=0)
        else:
            # Fallback: use brightest pixel
            max_idx = np.argmax(bright_map)
            h, w = bright_map.shape
            y, x = np.unravel_index(max_idx, (h, w))
            A = I_float[y, x]
        
        # Clamp to reasonable range
        A = np.clip(A, 0.2, 0.9)
        
        # Temporal smoothing with buffer
        if len(self.A_buffer) > 0:
            avg_A = np.mean(self.A_buffer, axis=0)
            A = 0.7 * A + 0.3 * avg_A  # Strong smoothing for stability
        
        return A
    
    def create_weight_map(self, dark_map, bright_map, A):
        """
        Create adaptive weight map based on darkness and atmosphere light
        """
        # Base weight from dark channel (darker = higher weight)
        weight = 1.0 - dark_map  # Invert: dark=1, bright=0
        
        # Adjust based on brightness difference from atmosphere
        avg_A = np.mean(A)
        brightness_diff = np.abs(bright_map - avg_A)
        
        # Areas very different from atmosphere light need more adjustment
        weight += 0.5 * brightness_diff
        
        # Reduce weight in already bright areas
        weight[bright_map > 0.7] *= 0.3
        
        # Normalize to [0, 1]
        weight = np.clip(weight, 0, 1)
        
        # Apply Gaussian blur for smooth transitions
        weight = cv2.GaussianBlur(weight, (5, 5), 1.0)
        
        return weight
    
    def adaptive_clahe_enhancement(self, image, weight_map, brightness_level):
        """
        Apply CLAHE with strength adapted by weight map
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Select CLAHE based on overall brightness
        if brightness_level < 0.25:
            clahe = self.clahe_levels['very_dark']
            base_boost = self.max_brightness_boost
        elif brightness_level < 0.45:
            clahe = self.clahe_levels['dark']
            base_boost = (self.max_brightness_boost + self.min_brightness_boost) / 2
        else:
            clahe = self.clahe_levels['medium']
            base_boost = self.min_brightness_boost
        
        # Apply CLAHE
        l_enhanced = clahe.apply(l)
        
        # Convert to float for weighted enhancement
        l_float = l_enhanced.astype(np.float32) / 255.0
        
        # Apply weight map for adaptive boosting
        if weight_map is not None:
            # Resize weight map if needed
            if weight_map.shape != l_float.shape:
                weight_map_resized = cv2.resize(weight_map, 
                                               (l_float.shape[1], l_float.shape[0]))
            else:
                weight_map_resized = weight_map
            
            # Create boost map: 1.0 to base_boost based on weight
            boost_map = 1.0 + (base_boost - 1.0) * weight_map_resized
            l_float = l_float * boost_map
        
        # Add detail enhancement
        if self.detail_strength > 0:
            l_blur = cv2.GaussianBlur(l_float, (3, 3), 0.5)
            detail = l_float - l_blur
            l_float = l_float + self.detail_strength * detail
        
        # Clip and convert back
        l_enhanced = np.clip(l_float * 255, 0, 255).astype(np.uint8)
        
        # Merge back (keep original color channels)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def color_correction_with_A(self, image, A):
        """
        Simple color correction using atmosphere light
        """
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Calculate average color of image
        avg_color = np.mean(img_float, axis=(0, 1))
        
        # Calculate color shift needed to match A
        color_shift = A - avg_color
        
        # Apply gentle color correction (30% of the shift)
        correction_strength = 0.3
        img_float += color_shift * correction_strength
        
        # Clip and convert back
        corrected = np.clip(img_float * 255, 0, 255).astype(np.uint8)
        
        return corrected
    
    def process_frame(self, frame):
        """
        Main processing pipeline for hybrid mode
        """
        start_time = time.perf_counter()
        
        # Store original for saving
        original_frame = frame.copy()
        
        # Resize for processing
        small = cv2.resize(frame, (self.proc_width, self.proc_height))
        
        # Convert to float for calculations
        I_float = small.astype(np.float32) / 255.0
        
        self.frame_count += 1
        
        # Calculate current brightness
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(gray) / 255.0
        self.brightness_buffer.append(current_brightness)
        
        # Check if we need full update
        need_full_update = (
            self.frame_count % self.update_rate == 0 or
            self.last_dark_map is None or
            len(self.brightness_buffer) < 5
        )
        
        if need_full_update:
            # Compute illumination channels
            dark_map, bright_map = self.compute_illumination_channels(I_float)
            
            # Compute atmosphere light (fast version)
            A = self.fast_atmosphere_light(I_float, bright_map)
            
            # Create weight map
            weight_map = self.create_weight_map(dark_map, bright_map, A)
            
            # Update state
            self.last_A = A
            self.last_dark_map = dark_map
            self.last_bright_map = bright_map
            self.last_weight_map = weight_map
            self.A_buffer.append(A)
            
            # Debug print (optional)
            if self.frame_count % 50 == 0:
                print(f"[Frame {self.frame_count}] A={A}, Brightness={current_brightness:.3f}")
        else:
            # Use cached values
            weight_map = self.last_weight_map
            A = self.last_A
        
        # Apply adaptive CLAHE enhancement
        avg_brightness = np.mean(self.brightness_buffer) if self.brightness_buffer else current_brightness
        enhanced_small = self.adaptive_clahe_enhancement(small, weight_map, avg_brightness)
        
        # Apply gentle color correction using A
        enhanced_small = self.color_correction_with_A(enhanced_small, self.last_A)
        
        # Resize back to display size
        enhanced = cv2.resize(enhanced_small, (self.display_width, self.display_height))
        
        # Calculate processing time
        proc_time = (time.perf_counter() - start_time) * 1000
        self.processing_times.append(proc_time)
        
        # Calculate FPS
        self.frame_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_counter / (current_time - self.last_fps_time)
            self.frame_counter = 0
            self.last_fps_time = current_time
        
        return enhanced, original_frame, proc_time, self.fps
    
    def save_comparison_images(self, enhanced, original):
        """Save both enhanced and original images for comparison"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create filenames
        enhanced_path = os.path.join(self.save_dir, f"enhanced_{timestamp}_{self.save_counter:04d}.jpg")
        original_path = os.path.join(self.save_dir, f"original_{timestamp}_{self.save_counter:04d}.jpg")
        
        # Save images
        cv2.imwrite(enhanced_path, enhanced)
        cv2.imwrite(original_path, original)
        
        print(f"Saved: {enhanced_path}")
        print(f"Saved: {original_path}")
        
        self.save_counter += 1
        
        return enhanced_path, original_path

# ==================== MAIN APPLICATION ====================

def setup_camera():
    """Configure IMX290 camera for low-light conditions"""
    picam2 = Picamera2()
    
    # Low-light optimized configuration
    config = picam2.create_preview_configuration(
        main={
            "size": (640, 480),
            "format": "RGB888"
        },
        controls={
            "FrameRate": 60,
            "ExposureTime": 40000,      # 40ms for better low-light capture
            "AnalogueGain": 2.5,        # Moderate gain
            "AwbEnable": True,          # Auto white balance
            "AeEnable": True,           # Auto exposure
            "Brightness": 0.1,          # Slightly brighter
            "Contrast": 1.1,            # Slightly more contrast
            "Saturation": 1.0,          # Neutral saturation (we'll adjust in code)
        }
    )
    
    picam2.configure(config)
    return picam2

def create_info_overlay(enhanced, fps, proc_time, frame_count, update_rate, A_value):
    """Create information overlay for display"""
    overlay = enhanced.copy()
    
    # Main info line
    info_line1 = f"FPS: {fps:.1f} | Time: {proc_time:.1f}ms | Frame: {frame_count}"
    cv2.putText(overlay, info_line1, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Secondary info
    info_line2 = f"Update: 1/{update_rate} | A: [{A_value[0]:.2f}, {A_value[1]:.2f}, {A_value[2]:.2f}]"
    cv2.putText(overlay, info_line2, (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Instructions
    instructions = "Q:Quit  S:Save  +:Brighter  -:Darker  U:Change Update Rate"
    cv2.putText(overlay, instructions, (10, 450),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return overlay

def main():
    """Main application loop"""
    print("=" * 70)
    print("HYBRID NIGHT VISION - Raspberry Pi 4 + IMX290")
    print("Combines Dark Channel Prior with Fast Atmosphere Light")
    print(f"Update Rate: 1/8 (reduced ghosting)")
    print("Display: horizontal flip so left/right matches physical")
    print("=" * 70)
    
    # Setup camera
    print("\nInitializing camera...")
    camera = setup_camera()
    camera.start()
    
    # Initialize enhancer with higher update rate (8 instead of 15)
    enhancer = HybridNightVision(
        width=640,
        height=480,
        patch_size=5,
        update_rate=8  # Higher frequency updates for less ghosting
    )
    
    # Create display window
    window_name = "Hybrid Night Vision"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    
    print("\nControls:")
    print("  Q - Quit application")
    print("  S - Save comparison images (original + enhanced)")
    print("  + - Increase brightness boost")
    print("  - - Decrease brightness boost")
    print("  U - Change update rate (8, 10, 15, 20)")
    print("  D - Toggle detail enhancement")
    print("  C - Toggle color correction")
    print()
    print("Images will be saved to 'comparison_images/' directory")
    print("\nStarting enhancement pipeline...")
    
    # Control variables
    detail_enabled = True
    color_correction_enabled = True
    current_update_rate = 8
    update_rates = [8, 10, 15, 20]
    
    try:
        while True:
            # Capture frame
            frame = camera.capture_array()
            # Fix mirror: so hand on physical left appears on screen left
            frame = cv2.flip(frame, 1)
            
            # Process frame
            enhanced, original, proc_time, fps = enhancer.process_frame(frame)
            
            # Create info overlay
            overlay = create_info_overlay(
                enhanced, fps, proc_time, 
                enhancer.frame_count, enhancer.update_rate,
                enhancer.last_A
            )
            
            # Display
            cv2.imshow(window_name, overlay)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
                
            elif key == ord('s'):
                # Save comparison images
                enhanced_path, original_path = enhancer.save_comparison_images(enhanced, original)
                
                # Show confirmation on screen for 1 second
                confirmation = "SAVED!"
                cv2.putText(overlay, confirmation, (300, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow(window_name, overlay)
                cv2.waitKey(100)  # Brief display
                
            elif key == ord('+'):
                # Increase brightness boost
                enhancer.max_brightness_boost = min(enhancer.max_brightness_boost + 0.1, 2.5)
                enhancer.min_brightness_boost = min(enhancer.min_brightness_boost + 0.05, 1.5)
                print(f"Brightness boost increased: Max={enhancer.max_brightness_boost:.1f}, Min={enhancer.min_brightness_boost:.1f}")
                
            elif key == ord('-'):
                # Decrease brightness boost
                enhancer.max_brightness_boost = max(enhancer.max_brightness_boost - 0.1, 1.2)
                enhancer.min_brightness_boost = max(enhancer.min_brightness_boost - 0.05, 1.0)
                print(f"Brightness boost decreased: Max={enhancer.max_brightness_boost:.1f}, Min={enhancer.min_brightness_boost:.1f}")
                
            elif key == ord('u'):
                # Cycle through update rates
                current_idx = update_rates.index(enhancer.update_rate)
                next_idx = (current_idx + 1) % len(update_rates)
                enhancer.update_rate = update_rates[next_idx]
                print(f"Update rate changed to: 1/{enhancer.update_rate}")
                
            elif key == ord('d'):
                # Toggle detail enhancement
                detail_enabled = not detail_enabled
                enhancer.detail_strength = 0.25 if detail_enabled else 0.0
                print(f"Detail enhancement: {'ON' if detail_enabled else 'OFF'}")
                
            elif key == ord('c'):
                # Toggle color correction
                color_correction_enabled = not color_correction_enabled
                # We'll handle this in the processing by skipping color correction step
                print(f"Color correction: {'ON' if color_correction_enabled else 'OFF'}")
                
            elif key == ord('i'):
                # Show/hide info
                pass  # Could implement toggle for info display
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\n" + "=" * 70)
        print("SESSION STATISTICS:")
        if enhancer.processing_times:
            avg_time = np.mean(enhancer.processing_times)
            avg_fps = 1000 / avg_time if avg_time > 0 else 0
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Average processing time: {avg_time:.1f}ms")
            print(f"  Total frames processed: {enhancer.frame_count}")
            print(f"  Images saved: {enhancer.save_counter}")
        
        print(f"\nImages saved in: {os.path.abspath(enhancer.save_dir)}")
        print("Goodbye!")

# ==================== PERFORMANCE MONITOR ====================

def run_performance_test():
    """Run a performance test without camera"""
    print("\n" + "=" * 70)
    print("PERFORMANCE TEST - HYBRID MODE")
    print("=" * 70)
    
    # Create test image
    test_image = np.random.randint(30, 100, (480, 640, 3), dtype=np.uint8)
    test_image = cv2.GaussianBlur(test_image, (7, 7), 1.5)
    
    # Test different update rates
    update_rates = [5, 8, 10, 15, 20]
    
    results = []
    
    for rate in update_rates:
        print(f"\nTesting update rate 1/{rate}...")
        
        enhancer = HybridNightVision(
            width=640,
            height=480,
            patch_size=5,
            update_rate=rate
        )
        
        # Warm up
        for _ in range(5):
            _, _, _, _ = enhancer.process_frame(test_image)
        
        # Benchmark
        times = []
        for i in range(30):
            start = time.perf_counter()
            _, _, _, _ = enhancer.process_frame(test_image)
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.mean(times)
        fps = 1000 / avg_time
        
        results.append((rate, fps, avg_time))
        
        print(f"  Avg time: {avg_time:.1f}ms")
        print(f"  FPS: {fps:.1f}")
    
    # Display results
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY (higher FPS is better):")
    for rate, fps, avg_time in results:
        print(f"  Update 1/{rate:2d}: {fps:5.1f} FPS ({avg_time:.1f}ms)")
    
    # Recommendation
    best = max(results, key=lambda x: x[1])
    print(f"\nRecommended update rate: 1/{best[0]} ({best[1]:.1f} FPS)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_performance_test()
    else:
        main()
