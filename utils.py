"""
Utility functions for scooter tracking system.
Includes: video I/O, coordinate transforms, speed estimation, logging.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
from collections import defaultdict, deque


def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO):
    """Configure logging with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


class VideoReader:
    """Wrapper for video reading with frame counting and FPS calculation."""
    
    def __init__(self, source: str):
        """
        Args:
            source: Path to video file or camera index (0, 1, etc.)
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame."""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
        return ret, frame
    
    def get_progress(self) -> float:
        """Get reading progress as percentage."""
        if self.frame_count > 0:
            return (self.current_frame / self.frame_count) * 100
        return 0.0
    
    def release(self):
        """Release video capture."""
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoWriter:
    """Wrapper for video writing with common codecs."""
    
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int],
                 codec: str = 'mp4v'):
        """
        Args:
            output_path: Output video file path
            fps: Frames per second
            frame_size: (width, height)
            codec: FourCC codec ('mp4v', 'x264', 'XVID', 'MJPG')
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        if not self.writer.isOpened():
            raise ValueError(f"Failed to open video writer: {output_path}")
    
    def write(self, frame: np.ndarray):
        """Write frame to video."""
        self.writer.write(frame)
    
    def release(self):
        """Release video writer."""
        self.writer.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class PixelToMeterConverter:
    """Convert pixel coordinates to real-world meters using scale or homography."""
    
    def __init__(self, scale: Optional[float] = None, 
                 homography_matrix: Optional[np.ndarray] = None):
        """
        Args:
            scale: Meters per pixel (e.g., 0.05 means 1 pixel = 0.05 meters)
            homography_matrix: 3x3 homography matrix for perspective transform
        """
        self.scale = scale
        self.homography = homography_matrix
        self.method = 'homography' if homography_matrix is not None else 'scale'
    
    def pixel_distance_to_meters(self, pixel_dist: float) -> float:
        """Convert pixel distance to meters using scale."""
        if self.scale is None:
            raise ValueError("Scale not set")
        return pixel_dist * self.scale
    
    def point_to_world(self, point: Tuple[int, int]) -> Tuple[float, float]:
        """Convert pixel point to world coordinates."""
        if self.method == 'homography':
            pt = np.array([[point[0], point[1]]], dtype=np.float32)
            pt = pt.reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(pt, self.homography)
            return transformed[0][0][0], transformed[0][0][1]
        else:
            # Simple scaling
            return point[0] * self.scale, point[1] * self.scale
    
    @staticmethod
    def compute_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        """
        Compute homography matrix from corresponding points.
        
        Args:
            src_points: Source points in image (Nx2)
            dst_points: Destination points in world coordinates (Nx2)
        
        Returns:
            3x3 homography matrix
        """
        H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
        return H


class SpeedEstimator:
    """Estimate object speed from tracking data."""
    
    def __init__(self, converter: PixelToMeterConverter, fps: float, 
                 smooth_window: int = 5):
        """
        Args:
            converter: PixelToMeterConverter instance
            fps: Video frame rate
            smooth_window: Number of frames for smoothing speed
        """
        self.converter = converter
        self.fps = fps
        self.smooth_window = smooth_window
        self.track_history = defaultdict(lambda: deque(maxlen=smooth_window))
        self.speed_history = defaultdict(lambda: deque(maxlen=smooth_window))
    
    def update(self, track_id: int, centroid: Tuple[int, int], 
               frame_idx: int) -> Optional[float]:
        """
        Update track and compute speed.
        
        Args:
            track_id: Unique track ID
            centroid: Current centroid (x, y)
            frame_idx: Current frame index
        
        Returns:
            Speed in m/s (None if insufficient data)
        """
        self.track_history[track_id].append((centroid, frame_idx))
        
        if len(self.track_history[track_id]) < 2:
            return None
        
        # Get two most recent points
        (prev_centroid, prev_frame), (curr_centroid, curr_frame) = \
            list(self.track_history[track_id])[-2:]
        
        # Compute pixel distance
        pixel_dist = np.linalg.norm(
            np.array(curr_centroid) - np.array(prev_centroid)
        )
        
        # Convert to meters
        meter_dist = self.converter.pixel_distance_to_meters(pixel_dist)
        
        # Compute time elapsed
        time_elapsed = (curr_frame - prev_frame) / self.fps
        
        if time_elapsed == 0:
            return None
        
        # Compute instantaneous speed
        speed = meter_dist / time_elapsed
        
        # Store and smooth
        self.speed_history[track_id].append(speed)
        smoothed_speed = np.mean(list(self.speed_history[track_id]))
        
        return smoothed_speed
    
    def get_speed_kmh(self, track_id: int) -> Optional[float]:
        """Get smoothed speed in km/h."""
        if track_id not in self.speed_history or len(self.speed_history[track_id]) == 0:
            return None
        speed_ms = np.mean(list(self.speed_history[track_id]))
        return speed_ms * 3.6  # Convert m/s to km/h


class Geofence:
    """Manage geofencing zones for violation detection."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Path to JSON config with zone definitions
        """
        self.zones = []
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load zones from JSON configuration."""
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.zones = config.get('zones', [])
    
    def add_zone(self, name: str, polygon: List[Tuple[int, int]], 
                 action: str = 'alert'):
        """
        Add a geofence zone.
        
        Args:
            name: Zone name
            polygon: List of (x, y) points defining polygon
            action: Action to take ('alert', 'log', 'ignore')
        """
        self.zones.append({
            'name': name,
            'polygon': np.array(polygon, dtype=np.int32),
            'action': action
        })
    
    def check_point(self, point: Tuple[int, int]) -> List[Dict]:
        """
        Check if point is inside any zone.
        
        Args:
            point: (x, y) coordinate
        
        Returns:
            List of violated zones
        """
        violations = []
        for zone in self.zones:
            result = cv2.pointPolygonTest(zone['polygon'], point, False)
            if result >= 0:  # Inside or on boundary
                violations.append(zone)
        return violations
    
    def draw_zones(self, frame: np.ndarray, color=(0, 255, 255), 
                   thickness: int = 2):
        """Draw all zones on frame."""
        for zone in self.zones:
            cv2.polylines(frame, [zone['polygon']], True, color, thickness)
            # Add zone name
            centroid = np.mean(zone['polygon'], axis=0).astype(int)
            cv2.putText(frame, zone['name'], tuple(centroid),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


class KalmanFilter:
    """Simple Kalman filter for tracking smoothing."""
    
    def __init__(self, process_noise: float = 1e-5, measurement_noise: float = 1e-1):
        """
        Args:
            process_noise: Process noise covariance (Q)
            measurement_noise: Measurement noise covariance (R)
        """
        # State: [x, y, vx, vy]
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
    
    def predict(self) -> Tuple[float, float]:
        """Predict next state."""
        prediction = self.kf.predict()
        return prediction[0, 0], prediction[1, 0]
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """Update with measurement."""
        measurement_array = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        corrected = self.kf.correct(measurement_array)
        return corrected[0, 0], corrected[1, 0]


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Compute intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    
    # Compute union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def create_directory_structure(base_path: str):
    """Create standard project directory structure."""
    dirs = [
        'models',
        'data/raw_videos',
        'data/frames',
        'data/labels/train',
        'data/labels/val',
        'data/labels/test',
        'output',
        'notebooks'
    ]
    
    base = Path(base_path)
    for d in dirs:
        (base / d).mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure at {base_path}")


if __name__ == "__main__":
    # Test utilities
    logger = setup_logger("test", "test.log")
    logger.info("Testing utilities module")
    
    # Test pixel to meter conversion
    converter = PixelToMeterConverter(scale=0.05)
    dist_m = converter.pixel_distance_to_meters(100)
    logger.info(f"100 pixels = {dist_m} meters")
    
    # Test IoU
    box1 = [10, 10, 50, 50]
    box2 = [30, 30, 70, 70]
    iou = compute_iou(box1, box2)
    logger.info(f"IoU between boxes: {iou:.3f}")
    
    print("Utils module test complete!")