"""
Lightweight scooter tracking pipeline using OpenCV background subtraction,
centroid tracking, and Kalman filtering. No deep learning required.
Suitable for CPU-only deployment and quick testing.
"""

import cv2
import numpy as np
from collections import OrderedDict, deque
import time
import argparse
from pathlib import Path
import csv
from typing import List, Tuple, Optional, Dict


class CentroidTracker:
    """
    Track objects using centroid distance matching.
    Based on simple Euclidean distance between centroids.
    """
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 50):
        """
        Args:
            max_disappeared: Max frames object can disappear before deregistering
            max_distance: Max distance for centroid matching (pixels)
        """
        self.next_object_id = 0
        self.objects = OrderedDict()  # id -> centroid
        self.disappeared = OrderedDict()  # id -> disappeared_count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Track history for trajectory visualization
        self.track_history = OrderedDict()  # id -> deque of centroids
    
    def register(self, centroid: Tuple[int, int]) -> int:
        """Register new object with unique ID."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.track_history[self.next_object_id] = deque(maxlen=30)
        self.track_history[self.next_object_id].append(centroid)
        
        obj_id = self.next_object_id
        self.next_object_id += 1
        return obj_id
    
    def deregister(self, object_id: int):
        """Remove object from tracking."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.track_history:
            del self.track_history[object_id]
    
    def update(self, rects: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int]]:
        """
        Update tracker with new detections.
        
        Args:
            rects: List of bounding boxes [(x1, y1, x2, y2), ...]
        
        Returns:
            Dictionary of {track_id: centroid}
        """
        # Handle no detections
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects
        
        # Compute centroids from bounding boxes
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)
        
        # If no objects exist, register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Match existing objects to new centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distance matrix
            D = np.linalg.norm(
                np.array(object_centroids)[:, None] - input_centroids[None, :],
                axis=2
            )
            
            # Find minimum distance matches
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows, used_cols = set(), set()
            
            for (r, c) in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                
                # Only match if distance is below threshold
                if D[r, c] > self.max_distance:
                    continue
                
                oid = object_ids[r]
                self.objects[oid] = input_centroids[c]
                self.disappeared[oid] = 0
                self.track_history[oid].append(tuple(input_centroids[c]))
                
                used_rows.add(r)
                used_cols.add(c)
            
            # Handle unmatched existing objects (disappeared)
            unused_rows = set(range(0, D.shape[0])) - used_rows
            for r in unused_rows:
                oid = object_ids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            
            # Register new objects
            unused_cols = set(range(0, D.shape[1])) - used_cols
            for c in unused_cols:
                self.register(input_centroids[c])
        
        return self.objects


class KalmanTracker:
    """Enhanced tracker with Kalman filtering for smoother predictions."""
    
    def __init__(self):
        self.trackers = {}  # track_id -> cv2.KalmanFilter
    
    def get_or_create(self, track_id: int) -> cv2.KalmanFilter:
        """Get existing Kalman filter or create new one."""
        if track_id not in self.trackers:
            kf = cv2.KalmanFilter(4, 2)  # 4 state (x,y,vx,vy), 2 measurements (x,y)
            kf.measurementMatrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=np.float32)
            
            kf.transitionMatrix = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
            
            self.trackers[track_id] = kf
        
        return self.trackers[track_id]
    
    def predict(self, track_id: int) -> Tuple[int, int]:
        """Predict next position."""
        kf = self.get_or_create(track_id)
        prediction = kf.predict()
        return int(prediction[0]), int(prediction[1])
    
    def update(self, track_id: int, measurement: Tuple[int, int]) -> Tuple[int, int]:
        """Update with measurement."""
        kf = self.get_or_create(track_id)
        measurement_array = np.array([[np.float32(measurement[0])],
                                     [np.float32(measurement[1])]])
        corrected = kf.correct(measurement_array)
        return int(corrected[0]), int(corrected[1])


class LightweightScooterTracker:
    """Complete lightweight tracking pipeline."""
    
    def __init__(self,
                 min_area: int = 500,
                 max_area: int = 50000,
                 min_aspect_ratio: float = 0.2,
                 max_aspect_ratio: float = 4.0,
                 mog2_history: int = 500,
                 mog2_threshold: float = 25,
                 max_disappeared: int = 40,
                 max_distance: float = 60):
        """
        Args:
            min_area: Minimum contour area (pixels)
            max_area: Maximum contour area (pixels)
            min_aspect_ratio: Minimum width/height ratio
            max_aspect_ratio: Maximum width/height ratio
            mog2_history: MOG2 history length
            mog2_threshold: MOG2 variance threshold
            max_disappeared: Max frames before deregistering track
            max_distance: Max centroid distance for matching
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_ar = min_aspect_ratio
        self.max_ar = max_aspect_ratio
        
        # Background subtractor
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=mog2_history,
            varThreshold=mog2_threshold,
            detectShadows=True
        )
        
        # Trackers
        self.centroid_tracker = CentroidTracker(max_disappeared, max_distance)
        self.kalman_tracker = KalmanTracker()
        
        # Morphological kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    def detect_objects(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect moving objects using background subtraction.
        
        Args:
            frame: Input BGR frame
        
        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fgmask = self.fgbg.apply(gray)
        
        # Threshold to remove shadows (value 127 in MOG2)
        _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel, iterations=2)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        rects = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(c)
            
            # Filter by aspect ratio
            ar = w / float(h) if h > 0 else 0
            if ar < self.min_ar or ar > self.max_ar:
                continue
            
            # Filter by minimum dimensions
            if h < 20 or w < 10:
                continue
            
            rects.append((x, y, x + w, y + h))
        
        return rects
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Process single frame through tracking pipeline.
        
        Args:
            frame: Input BGR frame
            frame_idx: Frame number
        
        Returns:
            Tuple of (annotated_frame, tracking_data)
        """
        # Detect objects
        rects = self.detect_objects(frame)
        
        # Update centroid tracker
        objects = self.centroid_tracker.update(rects)
        
        # Prepare tracking data
        tracking_data = {
            'frame_idx': frame_idx,
            'tracks': []
        }
        
        # Draw on frame
        annotated = frame.copy()
        
        # Draw bounding boxes
        for (x1, y1, x2, y2) in rects:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw tracks with Kalman smoothing
        for track_id, centroid in objects.items():
            # Update Kalman filter
            smoothed = self.kalman_tracker.update(track_id, centroid)
            
            # Draw smoothed centroid
            cv2.circle(annotated, smoothed, 5, (0, 255, 0), -1)
            
            # Draw ID
            text = f"ID {track_id}"
            cv2.putText(annotated, text, (smoothed[0] - 10, smoothed[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw trajectory
            if track_id in self.centroid_tracker.track_history:
                points = list(self.centroid_tracker.track_history[track_id])
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    cv2.line(annotated, points[i - 1], points[i], (0, 255, 255), 2)
            
            # Store track data
            tracking_data['tracks'].append({
                'track_id': track_id,
                'centroid': smoothed,
                'frame_idx': frame_idx
            })
        
        return annotated, tracking_data


def main():
    parser = argparse.ArgumentParser(description="Lightweight scooter tracker (CPU-only)")
    parser.add_argument('--video', type=str, default=None, help='Path to video file (None = webcam)')
    parser.add_argument('--output', type=str, default='output/lightweight_track.mp4',
                       help='Output video path')
    parser.add_argument('--csv', type=str, default='output/lightweight_tracks.csv',
                       help='Output CSV path')
    parser.add_argument('--display', action='store_true', help='Display tracking in real-time')
    parser.add_argument('--min-area', type=int, default=500, help='Minimum detection area')
    parser.add_argument('--max-area', type=int, default=50000, help='Maximum detection area')
    
    args = parser.parse_args()
    
    # Open video
    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        print(f"Error: Cannot open video source")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps:.2f} FPS")
    
    # Setup output video
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Setup CSV
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csvfile = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_idx', 'track_id', 'centroid_x', 'centroid_y', 'timestamp'])
    
    # Initialize tracker
    tracker = LightweightScooterTracker(
        min_area=args.min_area,
        max_area=args.max_area
    )
    
    print("Starting tracking... Press 'q' to quit")
    
    frame_idx = 0
    fps_calc = 0
    t0 = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated, tracking_data = tracker.process_frame(frame, frame_idx)
            
            # Calculate FPS
            elapsed = time.time() - t0
            if elapsed > 0:
                fps_calc = 0.9 * fps_calc + 0.1 * (1.0 / elapsed)
            t0 = time.time()
            
            # Add FPS text
            cv2.putText(annotated, f"FPS: {fps_calc:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add frame counter
            cv2.putText(annotated, f"Frame: {frame_idx}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add track count
            track_count = len(tracking_data['tracks'])
            cv2.putText(annotated, f"Tracks: {track_count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write to video
            out.write(annotated)
            
            # Write to CSV
            timestamp = frame_idx / fps
            for track in tracking_data['tracks']:
                csv_writer.writerow([
                    frame_idx,
                    track['track_id'],
                    track['centroid'][0],
                    track['centroid'][1],
                    timestamp
                ])
            
            # Display if requested
            if args.display:
                cv2.imshow("Lightweight Tracker", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        csvfile.close()
        cv2.destroyAllWindows()
        
        print(f"\nProcessed {frame_idx} frames")
        print(f"Output video: {output_path}")
        print(f"Output CSV: {csv_path}")


if __name__ == "__main__":
    main()