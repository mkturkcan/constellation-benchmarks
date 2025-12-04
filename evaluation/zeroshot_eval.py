"""
AutoModelForZeroShotObjectDetection Object Detection Evaluation on YOLO Dataset
Uses ultralytics metrics to compute precision, recall, and mAP
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


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


class ZeroShotDetector:
    """Wrapper for AutoModelForZeroShotObjectDetection model to detect objects"""
    
    def __init__(
        self, 
        class_mapping: Dict[str, int],
        model_id: str = "iSEE-Laboratory/llmdet_large",
        device: Optional[str] = None
    ):
        """
        Args:
            class_mapping: Dict mapping class names to YOLO label IDs
                          e.g., {'person': 1, 'vehicle': 0, 'bicycle': 1}
            model_id: HuggingFace model ID for the model
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.class_mapping = class_mapping
        self.model_id = model_id
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load LLMDet model and processor
        print(f"Loading LLMDet model ({model_id}) on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()  # Set to evaluation mode
        print("Model loaded successfully!")
        
        # Prepare text labels for detection
        # LLMDet expects labels in format like "a cat", "a car", etc.
        self.text_labels = [[f"a {class_name}" for class_name in self.class_mapping.keys()]]
        self.class_names_list = list(self.class_mapping.keys())
    
    def detect(
        self, 
        image: Image.Image, 
        confidence_threshold: float = 0.25
    ) -> List[Detection]:
        """
        Run detection on an image for all classes
        
        Args:
            image: PIL Image
            confidence_threshold: Minimum confidence threshold (default: 0.25)
            
        Returns:
            List of Detection objects with normalized coordinates
        """
        # Get image dimensions
        img_width, img_height = image.size
        
        # Prepare inputs
        inputs = self.processor(
            images=image, 
            text=self.text_labels, 
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process outputs
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=confidence_threshold,
            target_sizes=[(img_height, img_width)]
        )
        
        # Convert to Detection objects
        all_detections = []
        
        if len(results) > 0:
            result = results[0]
            
            for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                # Convert box from absolute to normalized coordinates
                # LLMDet returns [x_min, y_min, x_max, y_max] in absolute pixels
                x_min, y_min, x_max, y_max = box.tolist()
                
                # Normalize coordinates to [0, 1]
                norm_bbox = [
                    x_min / img_width,
                    y_min / img_height,
                    x_max / img_width,
                    y_max / img_height
                ]
                
                # Clip to valid range [0, 1]
                norm_bbox = [max(0.0, min(1.0, coord)) for coord in norm_bbox]
                
                # Map detected label back to YOLO class ID
                # Label is the detected class name (without "a " prefix)
                detected_class_name = label.strip()
                
                # Find matching class name in our mapping
                yolo_class_id = None
                for class_name, class_id in self.class_mapping.items():
                    # Check if the detected label matches (case-insensitive)
                    if class_name.lower() in detected_class_name.lower() or \
                       detected_class_name.lower() in class_name.lower():
                        yolo_class_id = class_id
                        break
                
                if yolo_class_id is not None:
                    all_detections.append(Detection(
                        class_id=yolo_class_id,
                        confidence=float(score.item()),
                        bbox=norm_bbox
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
        predictions: List of Detection objects
        ground_truths: List of GroundTruth objects
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
    
    # Sort predictions by confidence (already sorted in the calling code)
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


class ZeroShotYOLOEvaluator:
    """Main evaluator class"""
    
    def __init__(
        self,
        images_path: str,
        labels_path: str,
        class_mapping: Dict[str, int],
        class_names: Optional[Dict[int, str]] = None,
        model_id: str = "iSEE-Laboratory/llmdet_large",
        device: Optional[str] = None,
        confidence_threshold: float = 0.25
    ):
        """
        Args:
            images_path: Path to images directory
            labels_path: Path to labels directory
            class_mapping: Dict mapping class names to YOLO label IDs
            class_names: Optional dict mapping YOLO IDs to readable names
            model_id: HuggingFace model ID for LLMDet
            device: Device to run on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for detections
        """
        self.dataset = YOLODatasetLoader(images_path, labels_path)
        self.detector = ZeroShotDetector(class_mapping, model_id, device)
        self.class_mapping = class_mapping
        self.confidence_threshold = confidence_threshold
        
        # Create class names dict if not provided
        if class_names is None:
            # Infer from class_mapping (inverted)
            self.class_names = {}
            for name, class_id in class_mapping.items():
                if class_id not in self.class_names:
                    self.class_names[class_id] = name
        else:
            self.class_names = class_names
        
        # Statistics for metrics calculation
        self.stats = {
            'tp': [],           # True positives
            'conf': [],         # Confidence scores
            'pred_cls': [],     # Predicted classes
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
        
        # Process each image
        for img_idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
            image, ground_truths, image_name = self.dataset.get_item(img_idx)
            
            if verbose and img_idx % 10 == 0:
                print(f"\nProcessing image {img_idx}: {image_name}")
                print(f"  Ground truths: {len(ground_truths)}")
            
            # Run detection
            predictions = self.detector.detect(image, self.confidence_threshold)
            
            if verbose and img_idx % 10 == 0:
                print(f"  Predictions: {len(predictions)}")
            
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
        
        print(f"Total predictions: {len(stats_np['conf'])}")
        print(f"Total ground truths: {len(stats_np['target_cls'])}")
        print(f"TP array shape: {stats_np['tp'].shape}")
        
        # Use ultralytics ap_per_class function
        results = ap_per_class(
            stats_np['tp'],
            stats_np['conf'],
            stats_np['pred_cls'],
            stats_np['target_cls'],
            plot=False,
            names=self.class_names
        )
        
        # Unpack results
        tp, fp, p, r, f1, ap, unique_classes = results[:7]
        
        # Create metrics object
        det_metrics = DetMetrics(names=self.class_names)
        det_metrics.stats = {k: [v] for k, v in stats_np.items()}
        det_metrics.box.update(results[2:])
        det_metrics.box.nc = len(self.class_names)
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for i, class_id in enumerate(unique_classes):
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            per_class_metrics[class_name] = {
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
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nOverall Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  mAP@0.5:   {metrics['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<12}")
        print("-" * 68)
        
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"{class_name:<20} "
                  f"{class_metrics['precision']:<12.4f} "
                  f"{class_metrics['recall']:<12.4f} "
                  f"{class_metrics['ap50']:<12.4f} "
                  f"{class_metrics['ap50-95']:<12.4f}")
        
        print("="*60)


def main():
    """Example usage"""
    # Example configuration
    images_path = "/path/to/images"
    labels_path = "/path/to/labels"
    
    # Class mapping: detection class names -> YOLO label IDs
    class_mapping = {
        'person': 0,
        'car': 1,
        'bicycle': 2,
        # Add more classes as needed
    }
    
    # Optional: Provide readable class names
    class_names = {
        0: 'person',
        1: 'car',
        2: 'bicycle',
    }
    
    # Create evaluator
    evaluator = ZeroShotYOLOEvaluator(
        images_path=images_path,
        labels_path=labels_path,
        class_mapping=class_mapping,
        class_names=class_names,
        model_id="iSEE-Laboratory/llmdet_large",
        confidence_threshold=0.25
    )
    
    # Run evaluation
    metrics = evaluator.evaluate(verbose=True)
    
    # Print results
    evaluator.print_results(metrics)
    
    return metrics


if __name__ == "__main__":
    main()
