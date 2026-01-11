import cv2
import numpy as np
import math
import os
from Helper import KalmanTracker, CentroidTracker, contour_centroid

class Solution:
    def __init__(self):
        # Set to store unique IDs of vehicles that have been counted
        self.counted_ids = set()
        
    def determine_mode(self, cap):
        """
        Analyzes the first frame to decide if the video is 'Day' or 'Night'.
        It checks the borders of the image (ignoring the center) for:
        1. Low Color Saturation (typical of IR night cameras/fog).
        2. Low Brightness.
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video to start
        
        if not ret: return "day" # Default fallback
        
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # --- BORDER MASKING ---
        # Create a mask that covers only the outer 20% borders (Top, Bottom, Left, Right)
        # We ignore the center because headlights can make night look bright.
        mask = np.zeros((h, w), dtype="uint8")
        margin_x = int(w * 0.20)
        margin_y = int(h * 0.20)
        
        cv2.rectangle(mask, (0, 0), (w, margin_y), 255, -1)       # Top
        cv2.rectangle(mask, (0, h-margin_y), (w, h), 255, -1)     # Bottom
        cv2.rectangle(mask, (0, 0), (margin_x, h), 255, -1)       # Left
        cv2.rectangle(mask, (w-margin_x, 0), (w, h), 255, -1)     # Right
        
        # Calculate average saturation and brightness in the border regions
        saturation_mean = cv2.mean(hsv[..., 1], mask=mask)[0]
        brightness_mean = cv2.mean(hsv[..., 2], mask=mask)[0]
        
        # Logic: Low saturation (< 25) OR Low Brightness (< 80) = Night
        if saturation_mean < 25 or brightness_mean < 80:
            return "night"
        else:
            return "day"

    def adjust_gamma(self, image, gamma=1.0):
        """Adjusts image brightness (Gamma Correction). Used to brighten dark night videos."""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def apply_roi_mask(self, frame):
        """
        Applies a trapezoid mask to focus on the road.
        This cuts out the sky and distant trees which cause noise.
        """
        h, w = frame.shape[:2]
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        pts = np.array([
            [int(w * 0.20), int(h * 0.15)],  # Top Left
            [int(w * 0.85), int(h * 0.15)],  # Top Right
            [int(w * 0.95), h],              # Bottom Right
            [int(w * 0.10), h]               # Bottom Left
        ], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return cv2.bitwise_and(frame, frame, mask=mask)

    # ======================================================
    #                 NIGHT MODE ENGINE
    # ======================================================
    def run_night_engine(self, cap, w, h):
        # Parameters optimized for Night (Low light, noise)
        min_blob_area = 250 
        min_vehicle_area = 800
        trip_y = int(h * 0.55) # The virtual line vehicles must cross
        
        # Initialize Tracker and Background Subtractor
        tracker = KalmanTracker(max_disappeared=30, max_distance=150)
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=30, detectShadows=True)
        
        # Morphological Kernels (Shapes used to clean up noise)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15)) # Merges split headlights
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # --- 1. Preprocessing ---
            # Blur -> Mask ROI -> Boost Brightness (Gamma)
            frame_blurred = cv2.GaussianBlur(frame, (9, 9), 2)
            frame_masked = self.apply_roi_mask(frame_blurred)
            frame_processed = self.adjust_gamma(frame_masked, gamma=1.5)
            
            # --- 2. Object Detection ---
            fg = bg_subtractor.apply(frame_processed)
            _, fg = cv2.threshold(fg, 210, 255, cv2.THRESH_BINARY)
            
            # Clean up noise
            fg = cv2.erode(fg, kernel_erode, iterations=1)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_vertical, iterations=2)   
            fg = cv2.dilate(fg, kernel_dilate, iterations=1)

            # Find Contours (Blobs)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            current_centroids = []
            
            # --- 3. Filtering ---
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Ignore small noise blobs
                if area < min_blob_area: continue    
                if area < min_vehicle_area: continue 
                
                # Ignore very thin/tall lines (usually noise artifacts)
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw / float(bh) < 0.25: continue 
                
                cent = contour_centroid(cnt)
                if cent: current_centroids.append(cent)

            # --- 4. Tracking & Counting ---
            objects_coords, object_instances = tracker.update(current_centroids)

            for obj_id, (cx, cy) in objects_coords.items():
                kf_vehicle = object_instances[obj_id]
                
                # Check speed: Ignore stationary/jittering objects
                vx, vy = kf_vehicle.get_velocity()
                speed = math.hypot(vx, vy)
                if speed < 1.0: continue
                if kf_vehicle.age < 5: continue 

                # Tripwire Logic: Count if object crosses the line downwards
                if len(kf_vehicle.history) > 3:
                    prev_y = kf_vehicle.history[-3][1]
                    if prev_y < trip_y and cy > trip_y: 
                         if obj_id not in self.counted_ids:
                            self.counted_ids.add(obj_id)

    # ======================================================
    #                  DAY MODE ENGINE
    # ======================================================
    def run_day_engine(self, cap, w, h):
        # Parameters optimized for Day (Clear view, shadows)
        min_area = 800            
        trip_y = int(h * 0.55)
        buffer_y = trip_y - 20          
        border_margin = 50
        
        tracker = CentroidTracker(max_disappeared=20, max_distance=150)
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=60, detectShadows=True)
        
        start_positions = {} # To track where objects spawned
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # --- 1. Detection ---
            fg = bg_subtractor.apply(frame)
            
            # Anti-Flash: If >30% of screen changes at once (e.g., camera auto-adjust), skip frame
            if cv2.countNonZero(fg) > (w * h * 0.30): 
                continue 
                
            _, fg = cv2.threshold(fg, 240, 255, cv2.THRESH_BINARY)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_open, iterations=1)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            fg = cv2.dilate(fg, kernel_close, iterations=1)

            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            current_centroids = []

            # --- 2. Filtering ---
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area: continue
                
                # Solidity Check: Ignore "hollow" shapes (leaves/bushes)
                hull = cv2.convexHull(cnt)
                if cv2.contourArea(hull) > 0:
                    if float(area) / cv2.contourArea(hull) < 0.4: continue 

                cent = contour_centroid(cnt)
                if cent is not None: current_centroids.append(cent)

            # --- 3. Tracking ---
            objects = tracker.update(current_centroids)

            for object_id, (cx, cy) in objects.items():
                # Store start position to validate movement later
                if object_id not in start_positions:
                    # Only register start if near borders (entering frame)
                    in_margin = (cx < border_margin) or (cx > w - border_margin) or \
                                (cy < border_margin) or (cy > h - border_margin)
                    if in_margin: start_positions[object_id] = (cx, cy) 
                    else: start_positions[object_id] = None 
                
                if start_positions[object_id] is None: continue 

                # --- 4. Advanced Logic Filters ---
                frames_alive = len(tracker.history.get(object_id, []))
                start_x, start_y = start_positions[object_id]
                net_disp = math.hypot(start_x - cx, start_y - cy)
                
                # Stationary Filter: If alive long but moved little -> NOISE
                if frames_alive > 15 and net_disp < 30: continue 

                # Path Efficiency: Real cars move straight. Noise jitters.
                hist = tracker.history[object_id]
                total_path = sum([math.hypot(hist[i][0]-hist[i-1][0], hist[i][1]-hist[i-1][1]) for i in range(1, len(hist))])
                efficiency = net_disp / total_path if total_path > 0 else 0
                if frames_alive > 10 and efficiency < 0.35: continue 

                # --- 5. Counting ---
                should_count = False
                
                # Rule A: Buffer Zone Crossing (Standard)
                was_below = len(hist) >= 2 and hist[-2][1] > trip_y
                is_above_buffer = cy < buffer_y
                
                if was_below and is_above_buffer: should_count = True
                
                # Rule B: Pure Distance (For fast cars skipping frames)
                if net_disp > 100: should_count = True
                
                if should_count:
                    if object_id not in self.counted_ids:
                        self.counted_ids.add(object_id)

    def forward(self, video_path: str) -> int:
        """Main entry point required by the Hackathon."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return 0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        mode = self.determine_mode(cap)
        
        # Route to the correct engine based on Day/Night detection
        try:
            if mode == "night":
                self.run_night_engine(cap, w, h)
            else:
                self.run_day_engine(cap, w, h)
        except Exception as e:
            # Silently catch errors during processing to avoid crashing the whole pipeline
            print(f"Processing warning: {e}")
        finally:
            cap.release()
            
        return len(self.counted_ids)

# ======================================================
#            LOCAL TESTING BLOCK
# ======================================================
if __name__ == "__main__":
    solver = Solution()
    
    # 1. Ask user for input
    # input() takes the video path from the console
    input_video = input().strip() 
    
    # 2. Handle paths with quotes (e.g. "C:\Path\Video.mp4")
    if input_video.startswith('"') and input_video.endswith('"'):
        input_video = input_video[1:-1]
    
    # 3. Check and Run
    if not os.path.exists(input_video):
        print(f"Error: Video file '{input_video}' not found.")
    else:
        try:
            count = solver.forward(input_video)
            print(f"Final Count: {count}")
        except Exception as e:
            print(f"An error occurred: {e}")