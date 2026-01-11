import cv2
import numpy as np
import math
from collections import OrderedDict, deque

# ==========================================
#        HELPER FUNCTION
# ==========================================

def contour_centroid(contour):
    """
    Finds the center (x, y) of a blob(The blob of moving pixels representing a vehicle) based on its contour.
    """

    # Moments calculate the weighted average of pixel intensities
    M = cv2.moments(contour)
    
    # Safety check: If the blob has no area (it's a dot or noise), 
    # we can't divide by zero, so we skip it.
    
    if M["m00"] == 0: return None
    
    # Standard formula for centroid: (M10/Area, M01/Area)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

# ==========================================
#        TRACKER 1: KALMAN FILTER
#    (Best for Night Mode / Noisy Data)
# ==========================================
class KalmanVehicle:
    """
    This class represents 1 specific car being tracked.
    It uses a Kalman Filter to smooth out the path. If detection is lost, 
    it keeps moving the car based on its last known speed (prediction).
    """

    def __init__(self, initial_centroid):

        # We track 4 values (State): [x, y, velocity_x, velocity_y]
        # But we only measure 2 values: [x, y] from the camera.

        self.kf = cv2.KalmanFilter(4, 2)
        
        # 1. Measurement Matrix (H):
        # This tells the filter: "When we get a measurement, it adds a bijective map between x and y state."
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0], 
            [0, 1, 0, 0]
        ], np.float32)
        
        # 2. Transition Matrix (A): The Physics Engine
        # This matrix defines how the state changes from one frame to the next frame.
        # Logic: New_Pos = Old_Pos + Velocity
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x_new = x_old + vx
            [0, 1, 0, 1],  # y_new = y_old + vy
            [0, 0, 1, 0],  # vx stays constant (Simple Constant Velocity Model)
            [0, 0, 0, 1]   # vy stays constant
        ], np.float32)
        
        # 3. Process Noise (Q):
        # How much do we trust our physics model? 
        # Low values = "Trust the model, ignore jittery measurements."
        # This creates the smoothing effect essential for night tracking.
        self.kf.processNoiseCov = np.array([
            [1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 5, 0], 
            [0, 0, 0, 5]
        ], np.float32) * 0.03

        # Initialize the state with where we first saw the car
        self.kf.statePre = np.array([[initial_centroid[0]], [initial_centroid[1]], [0], [0]], np.float32)
        self.kf.statePost = np.array([[initial_centroid[0]], [initial_centroid[1]], [0], [0]], np.float32)
        
        # Keep a history trail (useful for drawing tails or debugging paths)
        self.history = deque(maxlen=20)
        self.history.append(initial_centroid)
        self.age = 0

    def predict(self):
        # "Guess" where the car will be in the next frame based on current velocity.
        prediction = self.kf.predict()
        return (int(prediction[0]), int(prediction[1]))

    def update(self, measurement):
        # "Correct" our guess with where we actually saw the car this frame.
        measured = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
        self.kf.correct(measured)
        self.history.append(measurement)
        self.age += 1

    def get_velocity(self):
        # Extract the calculated velocity from the internal state
        return (self.kf.statePost[2][0], self.kf.statePost[3][0])


class KalmanTracker:
    """
    The Manager Class.
    It maintains the list of all active KalmanVehicle objects.
    Its main job is the 'Association Problem': Figure out which new detection belongs to which existing car.
    """

    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_object_id = 0
        self.objects = OrderedDict() # Database of active cars {ID: KalmanVehicle}
        self.max_disappeared = max_disappeared # How long we keep a lost car before deleting it
        self.max_distance = max_distance # If a car 'jumps' further than this, it's probably a different car

    def register(self, centroid):
        # Create a new KalmanVehicle instance and assign it an ID
        self.objects[self.next_object_id] = KalmanVehicle(centroid)
        self.next_object_id += 1

    def update(self, input_centroids):
        output_coords = {}
        
        # Scenario 1: Clean slate. No cars tracked yet, so everything we see is new.
        if len(self.objects) == 0:
            for cent in input_centroids:
                self.register(cent)
        else:
            object_ids = list(self.objects.keys())
            # Ask every existing car to predict where it thinks it is right now
            object_centroids = [obj.predict() for obj in self.objects.values()]

            # Scenario 2: Cars exist, but we see nothing this frame.
            # We just let the cars 'age' without updating their position with new data.
            if len(input_centroids) == 0:
                for obj_id in object_ids:
                    self.objects[obj_id].age += 1 
            
            # Scenario 3: The hard part. Matching existing cars to new dots on screen.
            else:
                # Create a Distance Matrix (Every known car vs Every new dot)
                D = np.zeros((len(object_centroids), len(input_centroids)), dtype=float)
                for i, oc in enumerate(object_centroids):
                    for j, ic in enumerate(input_centroids):
                        D[i, j] = math.hypot(oc[0] - ic[0], oc[1] - ic[1])

                # Use a greedy approach to match pairs with the smallest distance first
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]
                used_rows, used_cols = set(), set()

                for row, col in zip(rows, cols):
                    # If this car or this input is already matched, skip
                    if row in used_rows or col in used_cols: continue
                    
                    # Sanity Check: If the closest dot is impossibly far away, 
                    # don't force a match. It's likely a new car, not the old one jumping.
                    if D[row, col] > self.max_distance: continue
                    
                    # It's a match! Update the Kalman filter with the real position
                    obj_id = object_ids[row]
                    self.objects[obj_id].update(input_centroids[col])
                    used_rows.add(row)
                    used_cols.add(col)
                
                # Anyone left over?
                # If an input dot wasn't matched to an existing car, it's a new car.
                unused_cols = set(range(0, D.shape[1])).difference(used_cols)
                for col in unused_cols:
                    self.register(input_centroids[col])

        # Return the positions to the main loop so we can draw them
        for obj_id, kf_vehicle in list(self.objects.items()):
            # We use the correct new position for display
            output_coords[obj_id] = (int(kf_vehicle.kf.statePost[0]), int(kf_vehicle.kf.statePost[1]))
            
        return output_coords, self.objects


# ==========================================
#         TRACKER 2: CENTROID TRACKER
#       (Best for Day Mode / Clear Data)
# ==========================================
class CentroidTracker:
    """
    A lightweight tracker. It assumes that the object in Frame 2 closest to 
    the object in Frame 1 is the same object. 
    Ideal for high-FPS, clean video where objects don't move much between frames.
    """
    
    def __init__(self, max_disappeared=15, max_distance=60):
        self.next_object_id = 0
        self.objects = OrderedDict()      # ID -> Coordinate Mapping
        self.disappeared = OrderedDict()  # ID -> How many frames since we last saw it
        self.history = OrderedDict()      # Stores the path for efficiency checks
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        # Start tracking a new object
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.history[self.next_object_id] = deque(maxlen=30)
        self.history[self.next_object_id].append(centroid)
        self.next_object_id += 1

    def deregister(self, object_id):
        # Stop tracking. Remove from all databases.
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.history[object_id]

    def update(self, input_centroids):
        # Scenario 1: We lost everything (e.g., empty road).
        # Mark all existing objects as 'missing' for one frame.
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                # If it's been missing too long, delete it.
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Scenario 2: First frame logic. Register everything we see.
        if len(self.objects) == 0:
            for cent in input_centroids:
                self.register(cent)
            return self.objects

        # Scenario 3: Mapping existing IDs to new coordinates.
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        
        # Calculate the Distance Matrix (Cost Matrix)
        # Rows = Existing IDs, Cols = New Inputs
        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=float)
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = math.hypot(oc[0] - ic[0], oc[1] - ic[1])

        # Sort to find the smallest distances first (Greedy Association)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            # If already matched, skip
            if row in used_rows or col in used_cols: continue
            
            # Distance Gating: If the jump is too big, it's not the same object.
            if D[row, col] > self.max_distance: continue
            
            # Match found! Update coordinates and reset the 'missing' counter.
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.history[object_id].append(input_centroids[col])
            self.disappeared[object_id] = 0
            
            used_rows.add(row)
            used_cols.add(col)

        # Cleanup: Check existing objects that weren't matched
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # New Objects: Check inputs that weren't matched to an ID
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects