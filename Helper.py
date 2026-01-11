import cv2
import numpy as np
import math
from collections import OrderedDict, deque

# ==========================================
#        HELPER FUNCTION
# ==========================================
def contour_centroid(contour):
    """
    Calculates the center (x, y) of a contour using moments.
    Returns None if the contour has zero area to avoid division by zero.
    """
    M = cv2.moments(contour)
    # m00 is the area. If area is 0, we can't find a center.
    if M["m00"] == 0: return None
    
    # Calculate centroid coordinates
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


# ==========================================
#        TRACKER 1: KALMAN FILTER
#    (Best for Night Mode / Noisy Data)
# ==========================================
class KalmanVehicle:
    """
    Represents a single vehicle being tracked using a Kalman Filter.
    The filter helps 'smooth' the path and predict position even if detection is lost briefly.
    """
    def __init__(self, initial_centroid):
        # Initialize OpenCV's Kalman Filter
        # 4 dynamic params (x, y, velocity_x, velocity_y)
        # 2 measured params (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # 1. Measurement Matrix (H): Relates state to measurement
        # We only measure position (x, y), not velocity.
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0], 
            [0, 1, 0, 0]
        ], np.float32)
        
        # 2. Transition Matrix (A): Physics model of movement
        # New_Pos = Old_Pos + Velocity * time
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx remains constant (simplified model)
            [0, 0, 0, 1]   # vy remains constant
        ], np.float32)
        
        # 3. Process Noise Covariance (Q): How much we trust our model vs reality
        # Small values mean we trust the model (smooths jitter).
        self.kf.processNoiseCov = np.array([
            [1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 5, 0], 
            [0, 0, 0, 5]
        ], np.float32) * 0.03

        # Initialize state with the first detection
        self.kf.statePre = np.array([[initial_centroid[0]], [initial_centroid[1]], [0], [0]], np.float32)
        self.kf.statePost = np.array([[initial_centroid[0]], [initial_centroid[1]], [0], [0]], np.float32)
        
        # History buffer to store past positions (useful for analyzing path)
        self.history = deque(maxlen=20)
        self.history.append(initial_centroid)
        self.age = 0

    def predict(self):
        """Estimates where the vehicle *should* be in the next frame."""
        prediction = self.kf.predict()
        return (int(prediction[0]), int(prediction[1]))

    def update(self, measurement):
        """Corrects the prediction using the actual detection from the current frame."""
        measured = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
        self.kf.correct(measured)
        self.history.append(measurement)
        self.age += 1

    def get_velocity(self):
        """Returns the estimated velocity (vx, vy)."""
        return (self.kf.statePost[2][0], self.kf.statePost[3][0])


class KalmanTracker:
    """
    Manages multiple KalmanVehicle instances.
    Responsible for assigning new detections to existing vehicles (Association Problem).
    """
    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_object_id = 0
        self.objects = OrderedDict() # Stores active vehicles: {ID: KalmanVehicle_Instance}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance # Max pixels a car can move between frames

    def register(self, centroid):
        """Adds a new vehicle to the tracker."""
        self.objects[self.next_object_id] = KalmanVehicle(centroid)
        self.next_object_id += 1

    def update(self, input_centroids):
        """
        Main logic: Matches input detections to existing tracked objects.
        Returns: Dictionary of {ID: (x, y)} for all active objects.
        """
        output_coords = {}
        
        # CASE 1: No objects currently tracking -> Register all inputs as new
        if len(self.objects) == 0:
            for cent in input_centroids:
                self.register(cent)
        else:
            object_ids = list(self.objects.keys())
            # Get predicted positions for all existing objects
            object_centroids = [obj.predict() for obj in self.objects.values()]

            # CASE 2: Objects exist but no inputs -> All objects age (predicted only)
            if len(input_centroids) == 0:
                for obj_id in object_ids:
                    self.objects[obj_id].age += 1 
            
            # CASE 3: We have objects AND inputs -> Match them!
            else:
                # Calculate distance matrix (Distance from every object to every input)
                D = np.zeros((len(object_centroids), len(input_centroids)), dtype=float)
                for i, oc in enumerate(object_centroids):
                    for j, ic in enumerate(input_centroids):
                        D[i, j] = math.hypot(oc[0] - ic[0], oc[1] - ic[1])

                # Find smallest distances to create pairs
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]
                used_rows, used_cols = set(), set()

                # Iterate over matched pairs
                for row, col in zip(rows, cols):
                    if row in used_rows or col in used_cols: continue
                    
                    # If distance is too large, it's not the same car -> Skip
                    if D[row, col] > self.max_distance: continue
                    
                    # Update the matched object with the new measurement
                    obj_id = object_ids[row]
                    self.objects[obj_id].update(input_centroids[col])
                    used_rows.add(row)
                    used_cols.add(col)
                
                # Any input that wasn't matched is a NEW object
                unused_cols = set(range(0, D.shape[1])).difference(used_cols)
                for col in unused_cols:
                    self.register(input_centroids[col])

        # Prepare output format for the main solution
        for obj_id, kf_vehicle in list(self.objects.items()):
            output_coords[obj_id] = (int(kf_vehicle.kf.statePost[0]), int(kf_vehicle.kf.statePost[1]))
            
        return output_coords, self.objects


# ==========================================
#        TRACKER 2: CENTROID TRACKER
#      (Best for Day Mode / Clear Data)
# ==========================================
class CentroidTracker:
    """
    A simpler, faster tracker based purely on Euclidean distance.
    Does not use Kalman filtering. Good for clean day footage.
    """
    def __init__(self, max_disappeared=15, max_distance=60):
        self.next_object_id = 0
        self.objects = OrderedDict()      # Stores {ID: (x, y)}
        self.disappeared = OrderedDict()  # Stores {ID: frames_lost_count}
        self.history = OrderedDict()      # Stores path history
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        """Registers a new object."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.history[self.next_object_id] = deque(maxlen=30)
        self.history[self.next_object_id].append(centroid)
        self.next_object_id += 1

    def deregister(self, object_id):
        """Removes an object that has been lost for too long."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.history[object_id]

    def update(self, input_centroids):
        """
        Matches detections to objects.
        Similar logic to KalmanTracker but updates position directly without prediction smoothing.
        """
        # CASE 1: No detections -> Increment 'disappeared' count for everyone
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # CASE 2: No tracked objects -> Register all inputs
        if len(self.objects) == 0:
            for cent in input_centroids:
                self.register(cent)
            return self.objects

        # CASE 3: Match existing objects to new inputs
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        
        # Compute Distance Matrix
        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=float)
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = math.hypot(oc[0] - ic[0], oc[1] - ic[1])

        # Match pairs based on minimum distance
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols: continue
            if D[row, col] > self.max_distance: continue
            
            # Update object position and reset disappeared counter
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.history[object_id].append(input_centroids[col])
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects (mark as disappeared)
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # Handle unmatched inputs (register as new objects)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects