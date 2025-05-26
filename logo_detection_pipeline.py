import cv2
import numpy as np
import torch
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
from PIL import Image
import requests
from transformers import pipeline, AutoProcessor, AutoModel
import clip
import argparse

@dataclass
class Detection:
    frame_idx: int
    timestamp: float
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    method: str

class LogoDetectionPipeline:
    def __init__(self, 
                 use_zero_shot: bool = True,
                 confidence_threshold: float = 0.5,
                 template_match_threshold: float = 0.7):
        
        self.confidence_threshold = confidence_threshold
        self.template_match_threshold = template_match_threshold
        self.use_zero_shot = use_zero_shot
        
        # Initialize detectors
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Initialize zero-shot models (lazy loading)
        self.clip_model = None
        self.clip_processor = None
        self.owlvit_detector = None
        
        # Tracking state
        self.active_trackers = []
        self.detection_history = []
        
    def _load_clip_model(self):
        """Lazy load CLIP model for feature similarity"""
        if self.clip_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_processor = clip.load("ViT-B/32", device=device)
            
    def _load_owlvit_model(self):
        """Lazy load OWL-ViT for zero-shot detection"""
        if self.owlvit_detector is None:
            self.owlvit_detector = pipeline(
                "zero-shot-object-detection",
                model="google/owlvit-base-patch32",
                device=0 if torch.cuda.is_available() else -1
            )
    
    def preprocess_video(self, video_path: str, 
                        max_fps: Optional[int] = None) -> List[Tuple[np.ndarray, float, int]]:
        """
        Preprocess video into frames with smart sampling
        Returns: List of (frame, timestamp, frame_idx)
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Smart frame sampling
        if max_fps and fps > max_fps:
            frame_step = int(fps / max_fps)
        else:
            frame_step = 1
            
        frames = []
        frame_idx = 0
        
        print(f"Processing video: {fps:.2f} FPS, {total_frames} frames")
        print(f"Sampling every {frame_step} frame(s)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_step == 0:
                timestamp = frame_idx / fps
                frames.append((frame, timestamp, frame_idx))
                
            frame_idx += 1
            
            if frame_idx % 1000 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        return frames
    
    def template_matching_orb(self, template: np.ndarray, 
                             frame: np.ndarray) -> Optional[Detection]:
        """Fast template matching using ORB features"""
        
        # Convert to grayscale
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Detect ORB features
        kp1, des1 = self.orb.detectAndCompute(template_gray, None)
        kp2, des2 = self.orb.detectAndCompute(frame_gray, None)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return None
            
        # Match features
        matches = self.bf_matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Filter good matches
        good_matches = [m for m in matches if m.distance < 50]  # Threshold
        
        if len(good_matches) < 10:
            return None
            
        # Calculate match quality
        match_ratio = len(good_matches) / len(matches) if matches else 0
        avg_distance = np.mean([m.distance for m in good_matches])
        confidence = match_ratio * (1.0 - avg_distance / 100.0)
        
        if confidence < self.template_match_threshold:
            return None
            
        # Estimate bounding box from matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                return None
                
            h, w = template_gray.shape
            pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            # Extract bounding box
            x_coords = dst[:, 0, 0]
            y_coords = dst[:, 0, 1]
            x, y, w, h = int(min(x_coords)), int(min(y_coords)), \
                        int(max(x_coords) - min(x_coords)), int(max(y_coords) - min(y_coords))
            
            return Detection(
                frame_idx=0,  # Will be set by caller
                timestamp=0.0,  # Will be set by caller
                confidence=confidence,
                bbox=(x, y, w, h),
                method="ORB"
            )
            
        except cv2.error:
            return None
    
    def zero_shot_detection(self, template: np.ndarray, 
                           frame: np.ndarray, 
                           logo_name: str) -> List[Detection]:
        """Zero-shot detection using OWL-ViT"""
        if not self.use_zero_shot:
            return []
            
        self._load_owlvit_model()
        
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Create text query
        queries = [f"a {logo_name} logo", f"{logo_name} brand", f"{logo_name}"]
        
        detections = []
        for query in queries:
            try:
                results = self.owlvit_detector(frame_pil, candidate_labels=[query])
                
                for result in results:
                    if result['score'] > self.confidence_threshold:
                        bbox = result['box']
                        x = int(bbox['xmin'])
                        y = int(bbox['ymin'])
                        w = int(bbox['xmax'] - bbox['xmin'])
                        h = int(bbox['ymax'] - bbox['ymin'])
                        
                        detection = Detection(
                            frame_idx=0,  # Will be set by caller
                            timestamp=0.0,  # Will be set by caller
                            confidence=result['score'],
                            bbox=(x, y, w, h),
                            method="OWL-ViT"
                        )
                        detections.append(detection)
                        
            except Exception as e:
                print(f"Zero-shot detection error: {e}")
                continue
                
        return detections
    
    def detect_logo_in_video(self, video_path: str, 
                            template_path: str, 
                            logo_name: str,
                            output_dir: str = "results") -> Dict:
        """
        Main detection pipeline
        """
        start_time = time.time()
        
        # Load template
        template = cv2.imread(template_path)
        if template is None:
            raise ValueError(f"Could not load template image: {template_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Preprocess video
        print("Preprocessing video...")
        frames = self.preprocess_video(video_path, max_fps=5)  # Sample at 5 FPS max
        
        all_detections = []
        processed_frames = 0
        
        print(f"Analyzing {len(frames)} frames...")
        
        for frame, timestamp, frame_idx in frames:
            frame_detections = []
            
            # Stage 1: Fast template matching
            orb_detection = self.template_matching_orb(template, frame)
            if orb_detection:
                orb_detection.frame_idx = frame_idx
                orb_detection.timestamp = timestamp
                frame_detections.append(orb_detection)
            
            # Stage 2: Zero-shot verification (if ORB found something or periodically)
            if self.use_zero_shot and (orb_detection or processed_frames % 50 == 0):
                zero_shot_detections = self.zero_shot_detection(template, frame, logo_name)
                for detection in zero_shot_detections:
                    detection.frame_idx = frame_idx
                    detection.timestamp = timestamp
                    frame_detections.append(detection)
            
            all_detections.extend(frame_detections)
            processed_frames += 1
            
            if processed_frames % 100 == 0:
                print(f"Processed {processed_frames}/{len(frames)} frames, found {len(all_detections)} detections")
        
        # Post-process detections
        filtered_detections = self._post_process_detections(all_detections)
        
        # Generate results
        results = {
            "video_path": video_path,
            "template_path": template_path,
            "logo_name": logo_name,
            "processing_time": time.time() - start_time,
            "total_frames_analyzed": len(frames),
            "total_detections": len(filtered_detections),
            "detections": [
                {
                    "frame_idx": d.frame_idx,
                    "timestamp": d.timestamp,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "method": d.method
                }
                for d in filtered_detections
            ]
        }
        
        # Save results
        results_path = os.path.join(output_dir, f"detections_{logo_name}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Detection complete! Found {len(filtered_detections)} logo instances")
        print(f"Results saved to: {results_path}")
        
        return results
    
    def _post_process_detections(self, detections: List[Detection]) -> List[Detection]:
        """Remove duplicates and filter low-confidence detections"""
        if not detections:
            return []
        
        # Sort by frame index
        detections.sort(key=lambda d: d.frame_idx)
        
        # Remove nearby duplicates (within 30 frames and similar bbox)
        filtered = []
        for detection in detections:
            is_duplicate = False
            for existing in filtered:
                if (abs(detection.frame_idx - existing.frame_idx) < 30 and
                    self._bbox_overlap(detection.bbox, existing.bbox) > 0.5):
                    # Keep the higher confidence detection
                    if detection.confidence > existing.confidence:
                        filtered.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def _bbox_overlap(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU overlap between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logo Detection Pipeline")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--template", required=True, help="Path to logo template image")
    parser.add_argument("--logo_name", required=True, help="Name of the logo (e.g., 'Nike', 'Kia')")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--use_zero_shot", action="store_true", help="Enable zero-shot detection")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = LogoDetectionPipeline(
        use_zero_shot=args.use_zero_shot,
        confidence_threshold=0.5,
        template_match_threshold=0.3
    )
    
    # Run detection
    results = pipeline.detect_logo_in_video(
        video_path=args.video,
        template_path=args.template,
        logo_name=args.logo_name,
        output_dir=args.output_dir
    )
    
    print(f"Detection completed in {results['processing_time']:.2f} seconds")
    print(f"Found {results['total_detections']} logo instances") 