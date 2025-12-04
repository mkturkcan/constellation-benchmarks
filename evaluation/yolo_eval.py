"""
YOLO Model Evaluation on YOLO Dataset
Evaluate a YOLO model on a target dataset with different class mappings
Uses ultralytics metrics to compute precision, recall, and mAP
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None


@dataclass
class Detection:
    """Represents a single detection"""
    class_id: int
    confidence: float
    bbox: List[float]  # [x_min, y_min, x_max, y_max] in normalized coordinates


@dataclass
class GroundTruth:
    """Represents a ground truth annotation"""
    class_id: int
    bbox: List[float]  # [x_min, y_min, x_max, y_max] in normalized coordinates


class YOLODatasetLoader:
    """Loads images and labels from YOLO dataset structure"""
    
    def __init__(self, images_path: str, labels_path: str):
        """
        Args:
            images_path: Path to directory containing images
            labels_path: Path to directory containing label .txt files
        """
        print(images_path, labels_path)
        self.images_path = Path(images_path)
        self.labels_path = Path(labels_path)
        
        # Get list of images
        self.image_files = sorted(list(self.images_path.glob("*.jpg")) + 
                                  list(self.images_path.glob("*.png")) +
                                  list(self.images_path.glob("*.jpeg")))
        
    def __len__(self):
        return len(self.image_files)
    
    def load_yolo_label(self, label_path: Path) -> List[GroundTruth]:
        """
        Load YOLO format labels from file.
        YOLO format: class_id x_center y_center width height (all normalized 0-1)
        
        Returns:
            List of GroundTruth objects with bboxes in [x_min, y_min, x_max, y_max] format
        """
        ground_truths = []
        
        if not label_path.exists():
            return ground_truths
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(round(float(parts[0])))
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert from center format to corner format
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                
                ground_truths.append(GroundTruth(
                    class_id=class_id,
                    bbox=[x_min, y_min, x_max, y_max]
                ))
        
        return ground_truths
    
    def get_item(self, idx: int) -> Tuple[Image.Image, List[GroundTruth], str]:
        """
        Get image and its ground truth annotations
        
        Returns:
            image: PIL Image
            ground_truths: List of GroundTruth objects
            image_name: Name of the image file
        """
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Get corresponding label file
        label_path = self.labels_path / f"{image_path.stem}.txt"
        ground_truths = self.load_yolo_label(label_path)
        
        return image, ground_truths, image_path.name


class YOLODetector:
    """Wrapper for YOLO model to detect objects"""
    
    def __init__(
        self, 
        model_path: str, 
        class_mapping: Dict[int, int],
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.45,
        device: str = None
    ):
        """
        Args:
            model_path: Path to YOLO model weights (e.g., 'yolov8n.pt')
            class_mapping: Dict mapping source model class IDs to target dataset class IDs
                          e.g., {0: 0, 1: 1, 2: 0} means source classes 0->target 0, 
                          1->target 1, 2->target 0
            conf_threshold: Confidence threshold for detections
            iou_threshold: NMS IoU threshold
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        if YOLO is None:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
        
        self.class_mapping = class_mapping
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load YOLO model
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        
        if device is not None:
            self.model.to(device)
        
        print(f"Model loaded successfully!")
        print(f"Model classes: {len(self.model.names)} classes")
        print(f"Source classes to evaluate: {sorted(class_mapping.keys())}")
        print(f"Target classes: {sorted(set(class_mapping.values()))}")
    
    def detect(self, image: Image.Image) -> List[Detection]:
        """
        Run detection on an image
        
        Args:
            image: PIL Image
            
        Returns:
            List of Detection objects with target class IDs
        """
        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        all_detections = []
        
        # Process results
        if len(results) > 0:
            result = results[0]
            
            # Get boxes, confidences, and classes
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxyn.cpu().numpy()  # Normalized xyxy format
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                # Get image dimensions for any needed conversions
                img_height, img_width = result.orig_shape
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    # Only include classes that are in our mapping
                    if cls in self.class_mapping:
                        # Map source class to target class
                        target_class = self.class_mapping[cls]
                        
                        # Box is already in normalized xyxy format
                        x_min, y_min, x_max, y_max = box
                        
                        all_detections.append(Detection(
                            class_id=target_class,
                            confidence=float(conf),
                            bbox=[float(x_min), float(y_min), float(x_max), float(y_max)]
                        ))
        
        return all_detections


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate IoU between two boxes in [x_min, y_min, x_max, y_max] format
    All coordinates should be normalized (0-1)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def match_predictions_to_ground_truth(
    predictions: List[Detection],
    ground_truths: List[GroundTruth],
    iou_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10)
) -> np.ndarray:
    """
    Match predictions to ground truth boxes and determine true positives
    
    Args:
        predictions: List of Detection objects (already mapped to target classes)
        ground_truths: List of GroundTruth objects (target dataset classes)
        iou_thresholds: Array of IoU thresholds to evaluate at
        
    Returns:
        tp: Boolean array of shape (num_predictions, num_iou_thresholds)
            True if prediction is a true positive at that IoU threshold
    """
    num_predictions = len(predictions)
    num_thresholds = len(iou_thresholds)
    
    # Initialize true positive array
    tp = np.zeros((num_predictions, num_thresholds), dtype=bool)
    
    if num_predictions == 0 or len(ground_truths) == 0:
        return tp
    
    # Group ground truths by class
    gt_by_class = {}
    for gt in ground_truths:
        if gt.class_id not in gt_by_class:
            gt_by_class[gt.class_id] = []
        gt_by_class[gt.class_id].append(gt)
    
    # Track which ground truths have been matched at each IoU threshold
    matched_gt = {threshold_idx: set() for threshold_idx in range(num_thresholds)}
    
    # Match each prediction to ground truth
    for pred_idx, pred in enumerate(predictions):
        if pred.class_id not in gt_by_class:
            continue
        
        # Get ground truths for this class
        class_gts = gt_by_class[pred.class_id]
        
        # Calculate IoU with all ground truths of the same class
        ious = [calculate_iou(pred.bbox, gt.bbox) for gt in class_gts]
        
        if len(ious) == 0:
            continue
        
        # Find the best matching ground truth
        best_gt_idx = np.argmax(ious)
        best_iou = ious[best_gt_idx]
        
        # Check if this is a true positive at each IoU threshold
        for threshold_idx, iou_threshold in enumerate(iou_thresholds):
            if best_iou >= iou_threshold:
                # Check if this ground truth hasn't been matched yet at this threshold
                gt_key = (pred.class_id, best_gt_idx)
                if gt_key not in matched_gt[threshold_idx]:
                    tp[pred_idx, threshold_idx] = True
                    matched_gt[threshold_idx].add(gt_key)
    
    return tp


class YOLOEvaluator:
    """Main evaluator class for YOLO model on target dataset"""
    
    def __init__(
        self,
        images_path: str,
        labels_path: str,
        model_path: str,
        class_mapping: Dict[int, int],
        target_class_names: Optional[Dict[int, str]] = None,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.45,
        device: str = None
    ):
        """
        Args:
            images_path: Path to images directory
            labels_path: Path to labels directory
            model_path: Path to YOLO model weights
            class_mapping: Dict mapping source model class IDs to target dataset class IDs
                          e.g., {0: 0, 1: 1, 2: 0}
            target_class_names: Optional dict mapping target class IDs to readable names
            conf_threshold: Confidence threshold for detections
            iou_threshold: NMS IoU threshold
            device: Device to run inference on
        """
        images_path = images_path[0]
        labels_path = labels_path[0]
        self.dataset = YOLODatasetLoader(images_path, labels_path)
        self.detector = YOLODetector(
            model_path, 
            class_mapping,
            conf_threshold,
            iou_threshold,
            device
        )
        self.class_mapping = class_mapping
        
        # Create target class names dict if not provided
        if target_class_names is None:
            # Get unique target classes
            target_classes = sorted(set(class_mapping.values()))
            self.target_class_names = {cls: f"class_{cls}" for cls in target_classes}
        else:
            self.target_class_names = target_class_names
        
        # Statistics for metrics calculation
        self.stats = {
            'tp': [],           # True positives
            'conf': [],         # Confidence scores
            'pred_cls': [],     # Predicted classes (target dataset classes)
            'target_cls': [],   # Target classes (ground truth)
            'target_img': []    # Image index for each ground truth
        }
    
    def evaluate(self, verbose: bool = True) -> Dict:
        """
        Run evaluation on entire dataset
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Starting evaluation on {len(self.dataset)} images...")
        print(f"Class mapping (source -> target): {self.class_mapping}")
        
        # Process each image
        for img_idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
            image, ground_truths, image_name = self.dataset.get_item(img_idx)
            
            if verbose and img_idx % 50 == 0 and img_idx > 0:
                print(f"\nProcessed {img_idx} images...")
                print(f"  Current image: {image_name}")
                print(f"  Total predictions so far: {len(self.stats['conf'])}")
                print(f"  Total ground truths so far: {len(self.stats['target_cls'])}")
            
            # Run detection
            predictions = self.detector.detect(image)
            
            # Sort predictions by confidence (descending)
            predictions.sort(key=lambda x: x.confidence, reverse=True)
            
            # Match predictions to ground truth
            if len(predictions) > 0:
                tp = match_predictions_to_ground_truth(predictions, ground_truths)
                
                # Update statistics
                self.stats['tp'].append(tp)
                self.stats['conf'].extend([pred.confidence for pred in predictions])
                self.stats['pred_cls'].extend([pred.class_id for pred in predictions])
            
            # Add ground truth statistics
            for gt in ground_truths:
                self.stats['target_cls'].append(gt.class_id)
                self.stats['target_img'].append(img_idx)
        
        # Calculate metrics
        print("\nCalculating metrics...")
        metrics = self.calculate_metrics()
        
        return metrics
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate precision, recall, and mAP using ultralytics method
        
        Returns:
            Dictionary containing metrics
        """
        from metrics_ultralytics import ap_per_class, DetMetrics
        
        # Convert lists to numpy arrays
        if len(self.stats['tp']) == 0:
            print("Warning: No predictions made!")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'mAP50': 0.0,
                'mAP50-95': 0.0,
                'per_class': {}
            }
        
        stats_np = {
            'tp': np.concatenate(self.stats['tp'], axis=0),
            'conf': np.array(self.stats['conf']),
            'pred_cls': np.array(self.stats['pred_cls']),
            'target_cls': np.array(self.stats['target_cls']),
            'target_img': np.array(self.stats['target_img'])
        }
        
        print(f"\nEvaluation Summary:")
        print(f"  Total predictions: {len(stats_np['conf'])}")
        print(f"  Total ground truths: {len(stats_np['target_cls'])}")
        print(f"  TP array shape: {stats_np['tp'].shape}")
        
        # Use ultralytics ap_per_class function
        results = ap_per_class(
            stats_np['tp'],
            stats_np['conf'],
            stats_np['pred_cls'],
            stats_np['target_cls'],
            plot=False,
            names=self.target_class_names
        )
        
        # Unpack results
        tp, fp, p, r, f1, ap, unique_classes = results[:7]
        
        # Create metrics object
        det_metrics = DetMetrics(names=self.target_class_names)
        det_metrics.stats = {k: [v] for k, v in stats_np.items()}
        det_metrics.box.update(results[2:])
        det_metrics.box.nc = len(self.target_class_names)
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for i, class_id in enumerate(unique_classes):
            class_name = self.target_class_names.get(class_id, f"class_{class_id}")
            per_class_metrics[class_name] = {
                'class_id': int(class_id),
                'precision': float(p[i]),
                'recall': float(r[i]),
                'f1': float(f1[i]),
                'ap50': float(ap[i, 0]),  # mAP@0.5
                'ap50-95': float(ap[i].mean())  # mAP@0.5:0.95
            }
        
        # Overall metrics
        metrics = {
            'precision': float(p.mean()) if len(p) > 0 else 0.0,
            'recall': float(r.mean()) if len(r) > 0 else 0.0,
            'mAP50': float(ap[:, 0].mean()) if len(ap) > 0 else 0.0,
            'mAP50-95': float(ap.mean()) if len(ap) > 0 else 0.0,
            'per_class': per_class_metrics
        }
        
        return metrics
    
    def print_results(self, metrics: Dict):
        """Print evaluation results in a readable format"""
        print("\n" + "="*70)
        print("YOLO MODEL EVALUATION RESULTS")
        print("="*70)
        
        print(f"\nClass Mapping (Source Model -> Target Dataset):")
        for src_cls, tgt_cls in sorted(self.class_mapping.items()):
            tgt_name = self.target_class_names.get(tgt_cls, f"class_{tgt_cls}")
            print(f"  Source class {src_cls} -> Target class {tgt_cls} ({tgt_name})")
        
        print(f"\nOverall Metrics:")
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")
        print(f"  mAP@0.5:      {metrics['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
        
        print(f"\nPer-Class Metrics (Target Dataset Classes):")
        print(f"{'Class':<20} {'ID':<5} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<12}")
        print("-" * 73)
        
        for class_name, class_metrics in sorted(metrics['per_class'].items()):
            print(f"{class_name:<20} "
                  f"{class_metrics['class_id']:<5} "
                  f"{class_metrics['precision']:<12.4f} "
                  f"{class_metrics['recall']:<12.4f} "
                  f"{class_metrics['ap50']:<12.4f} "
                  f"{class_metrics['ap50-95']:<12.4f}")
        
        print("="*70)


def main():
    """Example usage"""
    # Example configuration
    images_path = "/path/to/target/images"
    labels_path = "/path/to/target/labels"
    model_path = "yolov8n.pt"  # or path to your custom model
    
    # Class mapping: source model class IDs -> target dataset class IDs
    # Example: Your model has 80 COCO classes, but target dataset has only 3 classes
    class_mapping = {
        0: 0,   # Source class 0 (person in COCO) -> Target class 0
        1: 2,   # Source class 1 (bicycle in COCO) -> Target class 2
        2: 1,   # Source class 2 (car in COCO) -> Target class 1
        3: 1,   # Source class 3 (motorcycle) -> Target class 1 (vehicles)
        5: 1,   # Source class 5 (bus) -> Target class 1 (vehicles)
        7: 1,   # Source class 7 (truck) -> Target class 1 (vehicles)
        # Add more mappings as needed
    }
    
    # Target dataset class names (for better display)
    target_class_names = {
        0: 'person',
        1: 'vehicle',
        2: 'bicycle',
    }
    
    # Create evaluator
    evaluator = YOLOEvaluator(
        images_path=images_path,
        labels_path=labels_path,
        model_path=model_path,
        class_mapping=class_mapping,
        target_class_names=target_class_names,
        conf_threshold=0.001,
        iou_threshold=0.45,
        device='cuda'  # or 'cpu'
    )
    
    # Run evaluation
    metrics = evaluator.evaluate(verbose=True)
    
    # Print results
    evaluator.print_results(metrics)
    
    return metrics


if __name__ == "__main__":
    main()
