"""
Data collection module for extracting frames from videos and manual annotation.
Supports frame extraction at specified FPS and interactive bounding box drawing.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Optional
import json


class FrameExtractor:
    """Extract frames from video at specified FPS."""
    
    def __init__(self, video_path: str, output_dir: str, target_fps: Optional[float] = None):
        """
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted frames
            target_fps: Target FPS for extraction (None = use video FPS)
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.target_fps = target_fps if target_fps else self.video_fps
        self.frame_interval = int(self.video_fps / self.target_fps)
        
        self.metadata = []
        
        print(f"Video FPS: {self.video_fps:.2f}")
        print(f"Target extraction FPS: {self.target_fps:.2f}")
        print(f"Frame interval: {self.frame_interval}")
    
    def extract_frames(self, max_frames: Optional[int] = None) -> pd.DataFrame:
        """
        Extract frames and save metadata.
        
        Args:
            max_frames: Maximum number of frames to extract (None = all)
        
        Returns:
            DataFrame with frame metadata
        """
        frame_idx = 0
        extracted_count = 0
        
        video_name = Path(self.video_path).stem
        
        print(f"Extracting frames from {self.video_path}...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Extract frame at intervals
            if frame_idx % self.frame_interval == 0:
                frame_filename = f"{video_name}_frame_{extracted_count:06d}.jpg"
                frame_path = self.output_dir / frame_filename
                
                cv2.imwrite(str(frame_path), frame)
                
                timestamp = frame_idx / self.video_fps
                self.metadata.append({
                    'frame_id': extracted_count,
                    'frame_filename': frame_filename,
                    'video_frame_idx': frame_idx,
                    'timestamp': timestamp,
                    'source': self.video_path,
                    'width': frame.shape[1],
                    'height': frame.shape[0]
                })
                
                extracted_count += 1
                
                if extracted_count % 50 == 0:
                    print(f"Extracted {extracted_count} frames...")
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_idx += 1
        
        self.cap.release()
        
        # Save metadata to CSV
        df = pd.DataFrame(self.metadata)
        metadata_path = self.output_dir / f"{video_name}_metadata.csv"
        df.to_csv(metadata_path, index=False)
        
        print(f"\nExtracted {extracted_count} frames to {self.output_dir}")
        print(f"Metadata saved to {metadata_path}")
        
        return df


class InteractiveAnnotator:
    """Interactive tool for drawing bounding boxes on frames."""
    
    def __init__(self, frames_dir: str, output_dir: str):
        """
        Args:
            frames_dir: Directory containing extracted frames
            output_dir: Directory to save annotations
        """
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.frame_files = sorted(list(self.frames_dir.glob("*.jpg")))
        self.current_idx = 0
        
        self.drawing = False
        self.current_box = []
        self.boxes = []
        self.annotations = []
        
        print(f"Found {len(self.frame_files)} frames to annotate")
        print("\nControls:")
        print("  Left-click and drag: Draw bounding box")
        print("  SPACE: Save current frame annotations and move to next")
        print("  D: Delete last box")
        print("  C: Clear all boxes on current frame")
        print("  N: Next frame (without saving)")
        print("  P: Previous frame")
        print("  S: Save all annotations and quit")
        print("  Q: Quit without saving")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for drawing boxes."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [(x, y)]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box = [self.current_box[0], (x, y)]
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if len(self.current_box) == 2:
                self.boxes.append(self.current_box)
                self.current_box = []
    
    def draw_boxes(self, frame: np.ndarray) -> np.ndarray:
        """Draw all boxes on frame."""
        display = frame.copy()
        
        # Draw completed boxes
        for box in self.boxes:
            pt1, pt2 = box
            cv2.rectangle(display, pt1, pt2, (0, 255, 0), 2)
        
        # Draw current box being drawn
        if len(self.current_box) == 2:
            pt1, pt2 = self.current_box
            cv2.rectangle(display, pt1, pt2, (0, 255, 255), 2)
        
        # Show box count
        cv2.putText(display, f"Boxes: {len(self.boxes)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame info
        info_text = f"Frame {self.current_idx + 1}/{len(self.frame_files)}"
        cv2.putText(display, info_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display
    
    def save_frame_annotations(self):
        """Save annotations for current frame."""
        if len(self.boxes) == 0:
            return
        
        frame_file = self.frame_files[self.current_idx]
        frame_name = frame_file.stem
        
        for box_idx, box in enumerate(self.boxes):
            pt1, pt2 = box
            x1, y1 = pt1
            x2, y2 = pt2
            
            # Ensure coordinates are ordered
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            
            self.annotations.append({
                'frame_filename': frame_file.name,
                'frame_path': str(frame_file),
                'box_id': box_idx,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'width': x_max - x_min,
                'height': y_max - y_min,
                'class': 'scooter',
                'timestamp': datetime.now().isoformat()
            })
        
        print(f"Saved {len(self.boxes)} boxes for {frame_file.name}")
        self.boxes = []
    
    def run(self):
        """Run interactive annotation loop."""
        if len(self.frame_files) == 0:
            print("No frames found!")
            return
        
        cv2.namedWindow("Annotator")
        cv2.setMouseCallback("Annotator", self.mouse_callback)
        
        while True:
            if self.current_idx >= len(self.frame_files):
                print("Reached end of frames")
                break
            
            frame_path = self.frame_files[self.current_idx]
            frame = cv2.imread(str(frame_path))
            
            if frame is None:
                print(f"Failed to load {frame_path}")
                self.current_idx += 1
                continue
            
            display = self.draw_boxes(frame)
            cv2.imshow("Annotator", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting without saving...")
                break
            
            elif key == ord('s'):
                self.save_frame_annotations()
                self.save_all_annotations()
                print("Saved all annotations!")
                break
            
            elif key == ord(' '):
                self.save_frame_annotations()
                self.current_idx += 1
            
            elif key == ord('n'):
                self.current_idx += 1
                self.boxes = []
            
            elif key == ord('p'):
                if self.current_idx > 0:
                    self.current_idx -= 1
                self.boxes = []
            
            elif key == ord('d'):
                if len(self.boxes) > 0:
                    self.boxes.pop()
                    print("Deleted last box")
            
            elif key == ord('c'):
                self.boxes = []
                print("Cleared all boxes")
        
        cv2.destroyAllWindows()
    
    def save_all_annotations(self):
        """Save all annotations to CSV and JSON."""
        if len(self.annotations) == 0:
            print("No annotations to save")
            return
        
        # Save as CSV
        df = pd.DataFrame(self.annotations)
        csv_path = self.output_dir / "annotations.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV annotations to {csv_path}")
        
        # Save as JSON
        json_path = self.output_dir / "annotations.json"
        with open(json_path, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        print(f"Saved JSON annotations to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames and annotate scooters")
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--fps', type=float, default=5.0, help='Target FPS for extraction')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames to extract')
    parser.add_argument('--annotate', action='store_true', help='Launch interactive annotator')
    
    args = parser.parse_args()
    
    # Extract frames
    extractor = FrameExtractor(args.video, args.output, args.fps)
    metadata_df = extractor.extract_frames(args.max_frames)
    
    # Launch annotator if requested
    if args.annotate:
        print("\nLaunching interactive annotator...")
        annotator = InteractiveAnnotator(args.output, args.output)
        annotator.run()


if __name__ == "__main__":
    main()