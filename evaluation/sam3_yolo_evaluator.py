"""
SAM3 Model Evaluation on YOLO Dataset with Visualization
Calculates precision, recall, and mAP using ultralytics metrics approach
Visualizes predictions and ground truth on images
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Import metrics functions
from metrics import box_iou, ap_per_class

# Import visualization
from visualization import BoundingBoxVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLODataset:
    """Handle YOLO dataset structure with images and labels"""
    
    def __init__(self, images_path: str, labels_path: str):
        """
        Initialize YOLO dataset loader
        
        Args:
            images_path: Path to directory containing images
            labels_path: Path to directory containing label .txt files
        """
        self.images_path = Path(images_path)
        self.labels_path = Path(labels_path)
        
        # Get all image files
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_files = []
        for ext in self.image_extensions:
            self.image_files.extend(list(self.images_path.glob(f'*{ext}')))
            self.image_files.extend(list(self.images_path.glob(f'*{ext.upper()}')))
        
        self.image_files = sorted(self.image_files)
        logger.info(f"Found {len(self.image_files)} images in {images_path}")
    
    def load_label(self, image_path: Path) -> np.ndarray:
        """
        Load YOLO format label file for an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy array of shape (N, 5) where each row is [class_id, x_center, y_center, width, height]
            All coordinates are normalized (0-1)
        """
        label_path = self.labels_path / (image_path.stem + '.txt')
        
        if not label_path.exists():
            return np.zeros((0, 5))  # No labels for this image
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    # class_id, x_center, y_center, width, height
                    labels.append([float(x) for x in parts[:5]])
        
        return np.array(labels) if labels else np.zeros((0, 5))
    
    def yolo_to_xyxy(self, boxes: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
        """
        Convert YOLO format boxes (normalized x_center, y_center, width, height) 
        to xyxy format (x1, y1, x2, y2) in pixel coordinates
        
        Args:
            boxes: Array of shape (N, 4+) with YOLO format boxes in columns 1-4
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Array of shape (N, 4) with boxes in xyxy format
        """
        if len(boxes) == 0:
            return np.zeros((0, 4))
        
        boxes_xyxy = np.zeros((len(boxes), 4))
        
        # Extract normalized coordinates
        x_center = boxes[:, 1] * img_width
        y_center = boxes[:, 2] * img_height
        width = boxes[:, 3] * img_width
        height = boxes[:, 4] * img_height
        
        # Convert to xyxy
        boxes_xyxy[:, 0] = x_center - width / 2   # x1
        boxes_xyxy[:, 1] = y_center - height / 2  # y1
        boxes_xyxy[:, 2] = x_center + width / 2   # x2
        boxes_xyxy[:, 3] = y_center + height / 2  # y2
        
        return boxes_xyxy
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, np.ndarray, Path]:
        """
        Get image and labels at index
        
        Returns:
            image: PIL Image
            labels: numpy array of shape (N, 5) with [class_id, x_center, y_center, width, height]
            image_path: Path to image file
        """
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        labels = self.load_label(image_path)
        return image, labels, image_path


class SAM3Evaluator:
    """Evaluate SAM3 model on YOLO dataset with ultralytics metrics and visualization"""
    
    def __init__(
        self,
        class_mapping: Dict[str, int],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        iou_thresholds: Optional[List[float]] = None,
        conf_threshold: float = 0.001,
        visualize: bool = False,
        vis_output_dir: Optional[str] = None,
        vis_mode: str = 'both'
    ):
        """
        Initialize SAM3 Evaluator
        
        Args:
            class_mapping: Dictionary mapping SAM3 class names to YOLO class IDs
                          e.g., {'person': 1, 'vehicle': 0, 'bike': 2}
            device: Device to run model on ('cuda' or 'cpu')
            iou_thresholds: IoU thresholds for mAP calculation (default: 0.5 to 0.95 step 0.05)
            conf_threshold: Confidence threshold for predictions
            visualize: Whether to save visualizations
            vis_output_dir: Directory to save visualizations (default: './visualizations')
            vis_mode: Visualization mode - 'predictions', 'ground_truth', 'both', 'matches', 'comparison'
        """
        self.class_mapping = class_mapping
        self.device = device
        self.conf_threshold = conf_threshold
        self.visualize = visualize
        self.vis_mode = vis_mode
        
        # IoU thresholds for mAP (COCO-style: 0.5:0.95)
        if iou_thresholds is None:
            self.iou_thresholds = np.linspace(0.5, 0.95, 10)
        else:
            self.iou_thresholds = np.array(iou_thresholds)
        
        # Initialize SAM3 model
        logger.info("Loading SAM3 model...")
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        logger.info("SAM3 model loaded successfully")
        
        # Storage for evaluation statistics
        self.stats = {
            'tp': [],        # True positives
            'conf': [],      # Confidence scores
            'pred_cls': [],  # Predicted classes
            'target_cls': [], # Target classes
            'target_img': []  # Target image indices
        }
        
        # Get unique class IDs and create reverse mapping
        self.yolo_class_ids = sorted(set(class_mapping.values()))
        self.class_names = {v: k for k, v in class_mapping.items()}
        self.nc = len(self.yolo_class_ids)  # Number of classes
        
        # Initialize visualizer if needed
        if self.visualize:
            if vis_output_dir is None:
                vis_output_dir = './visualizations'
            self.visualizer = BoundingBoxVisualizer(
                output_dir=vis_output_dir,
                show_labels=True,
                show_confidence=True
            )
            logger.info(f"Visualization enabled. Output: {vis_output_dir}")
            
            # Create legend
            legend = self.visualizer.create_legend()
            self.visualizer.save_image(legend, 'legend.png')
        else:
            self.visualizer = None
    
    def predict(self, image: Image.Image, class_name: str) -> Dict:
        """
        Run SAM3 prediction on image for a specific class
        
        Args:
            image: PIL Image
            class_name: Name of class to detect (e.g., 'person', 'vehicle')
            
        Returns:
            Dictionary with 'boxes', 'scores', 'masks'
        """
        # Set image in processor
        inference_state = self.processor.set_image(image)
        
        # Prompt with class name
        output = self.processor.set_text_prompt(state=inference_state, prompt=class_name)
        
        return {
            'boxes': output['boxes'].cpu(),  # Shape: (N, 4) in xyxy format
            'scores': output['scores'].cpu(),  # Shape: (N,)
            'masks': output['masks'].cpu()     # Shape: (N, H, W)
        }
    
    def match_predictions(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        iou_thresholds: np.ndarray
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Match predictions to targets using IoU
        
        Args:
            pred_boxes: Predicted boxes in xyxy format (N, 4)
            target_boxes: Target boxes in xyxy format (M, 4)
            iou_thresholds: Array of IoU thresholds
            
        Returns:
            correct: Array of shape (N, len(iou_thresholds)) with True/False for each prediction
            matched_gt_indices: List of matched ground truth indices
        """
        if len(pred_boxes) == 0 or len(target_boxes) == 0:
            return np.zeros((len(pred_boxes), len(iou_thresholds)), dtype=bool), []
        
        # Calculate IoU matrix: (N, M)
        iou = box_iou(pred_boxes, target_boxes).cpu().numpy()
        
        # For each prediction, find best matching target
        correct = np.zeros((len(pred_boxes), len(iou_thresholds)), dtype=bool)
        matched_gt_indices = []
        
        # Get best IoU for each prediction
        if len(target_boxes) > 0:
            # For each prediction, get the best matching target
            matches = iou.argmax(axis=1)  # Best target for each prediction
            best_iou = iou[np.arange(len(pred_boxes)), matches]
            
            # Mark as correct at each IoU threshold
            for i, threshold in enumerate(iou_thresholds):
                correct[:, i] = best_iou >= threshold
            
            # Store matched indices (at IoU 0.5)
            matched_gt_indices = [int(matches[i]) for i, iou_val in enumerate(best_iou) if iou_val >= 0.25]
        
        return correct, matched_gt_indices
    
    def visualize_image_predictions(
        self,
        image: Image.Image,
        all_predictions: Dict[int, Dict],
        labels: np.ndarray,
        image_name: str,
        image_idx: int
    ):
        """
        Visualize all predictions and ground truth for an image
        
        Args:
            image: PIL Image
            all_predictions: Dict mapping class_id to predictions dict
            labels: Ground truth labels
            image_name: Name for output file
            image_idx: Image index
        """
        if not self.visualize or self.visualizer is None:
            return
        
        img_width, img_height = image.size
        
        # Prepare combined predictions for all classes
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_classes = []
        
        for class_id, preds in all_predictions.items():
            if len(preds['boxes']) > 0:
                for box, score in zip(preds['boxes'], preds['scores']):
                    all_pred_boxes.append(box)
                    all_pred_scores.append(score)
                    all_pred_classes.append(class_id)
        
        # Create combined predictions dict
        combined_preds = {
            'boxes': all_pred_boxes,
            'scores': all_pred_scores,
            'classes': all_pred_classes
        }
        
        if self.vis_mode == 'comparison':
            # Create side-by-side comparison
            vis_image = self.visualizer.create_comparison(
                image, combined_preds, labels, self.class_names, image_name
            )
            output_path = self.visualizer.save_image(
                vis_image, f"{image_idx:06d}_{image_name}_comparison.jpg"
            )
        else:
            # Standard visualization
            vis_image = image.copy()
            from PIL import ImageDraw
            draw = ImageDraw.Draw(vis_image)
            
            # Draw ground truth
            if self.vis_mode in ['ground_truth', 'both'] and len(labels) > 0:
                from visualization import Colors
                for gt_box in labels:
                    class_id = int(gt_box[0])
                    x_center, y_center, width, height = gt_box[1:5]
                    x1 = (x_center - width / 2) * img_width
                    y1 = (y_center - height / 2) * img_height
                    x2 = (x_center + width / 2) * img_width
                    y2 = (y_center + height / 2) * img_height
                    
                    box = [x1, y1, x2, y2]
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    label = f"GT: {class_name}"
                    color = Colors.GROUND_TRUTH
                    self.visualizer.draw_box(draw, box, color, label)
            
            # Draw predictions
            if self.vis_mode in ['predictions', 'both'] and len(all_pred_boxes) > 0:
                from visualization import Colors
                for box, score, class_id in zip(all_pred_boxes, all_pred_scores, all_pred_classes):
                    box_list = box.cpu().numpy().tolist() if hasattr(box, 'cpu') else box
                    score_val = score.item() if hasattr(score, 'item') else score
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    label = f"{class_name}: {score_val:.2f}"
                    color = Colors.PREDICTION
                    self.visualizer.draw_box(draw, box_list, color, label)
            
            output_path = self.visualizer.save_image(
                vis_image, f"{image_idx:06d}_{image_name}"
            )
        
        logger.debug(f"Saved visualization: {output_path}")
    
    def evaluate_dataset(
        self,
        dataset: YOLODataset,
        batch_size: int = 1,
        save_vis_every: int = 1
    ) -> Dict:
        """
        Evaluate model on entire dataset
        
        Args:
            dataset: YOLODataset instance
            batch_size: Batch size (currently only supports 1)
            save_vis_every: Save visualization every N images (1 = all images)
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating on {len(dataset)} images...")
        
        # Clear statistics
        self.stats = {
            'tp': [],
            'conf': [],
            'pred_cls': [],
            'target_cls': [],
            'target_img': []
        }
        
        # Process each image
        for img_idx in tqdm(range(len(dataset)), desc="Processing images"):
            image, labels, image_path = dataset[img_idx]
            img_width, img_height = image.size
            
            # Store all predictions for this image (for visualization)
            all_predictions = {}
            
            # Get predictions for each class
            for class_name, yolo_class_id in self.class_mapping.items():
                # Run prediction
                predictions = self.predict(image, class_name)
                
                pred_boxes = predictions['boxes']
                pred_scores = predictions['scores']
                
                # Filter by confidence
                conf_mask = pred_scores >= self.conf_threshold
                pred_boxes = pred_boxes[conf_mask]
                pred_scores = pred_scores[conf_mask]
                
                n_pred = len(pred_boxes)
                
                # Store for visualization
                all_predictions[yolo_class_id] = {
                    'boxes': pred_boxes,
                    'scores': pred_scores,
                    'class_id': yolo_class_id
                }
                
                # Get targets for this class
                if len(labels) > 0:
                    target_mask = labels[:, 0] == yolo_class_id
                    class_targets = labels[target_mask]
                else:
                    class_targets = np.zeros((0, 5))
                
                n_target = len(class_targets)
                
                # Convert target boxes to xyxy
                if n_target > 0:
                    target_boxes_xyxy = dataset.yolo_to_xyxy(
                        class_targets, img_width, img_height
                    )
                    target_boxes_xyxy = torch.from_numpy(target_boxes_xyxy).float()
                else:
                    target_boxes_xyxy = torch.zeros((0, 4))
                
                # Match predictions to targets
                if n_pred > 0:
                    if n_target > 0:
                        # Calculate IoU and match
                        correct, matched_indices = self.match_predictions(
                            pred_boxes, target_boxes_xyxy, self.iou_thresholds
                        )
                    else:
                        # No targets, all predictions are false positives
                        correct = np.zeros((n_pred, len(self.iou_thresholds)), dtype=bool)
                        matched_indices = []
                    
                    # Store statistics
                    self.stats['tp'].append(correct)
                    self.stats['conf'].append(pred_scores.numpy())
                    self.stats['pred_cls'].append(np.full(n_pred, yolo_class_id))
                
                # Store target information
                if n_target > 0:
                    self.stats['target_cls'].append(np.full(n_target, yolo_class_id))
                    self.stats['target_img'].append(np.full(n_target, img_idx))
            
            # Visualize this image
            if self.visualize and (img_idx % save_vis_every == 0):
                self.visualize_image_predictions(
                    image, all_predictions, labels,
                    image_path.name, img_idx
                )
        
        # Compute metrics
        return self.compute_metrics()
    
    def compute_metrics(self) -> Dict:
        """
        Compute precision, recall, mAP using ultralytics ap_per_class function
        
        Returns:
            Dictionary with metrics
        """
        # Concatenate all statistics
        if not self.stats['tp']:
            logger.warning("No predictions made!")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'mAP50': 0.0,
                'mAP50-95': 0.0,
                'per_class': {}
            }
        
        stats = {
            'tp': np.concatenate(self.stats['tp'], 0),
            'conf': np.concatenate(self.stats['conf'], 0),
            'pred_cls': np.concatenate(self.stats['pred_cls'], 0),
            'target_cls': np.concatenate(self.stats['target_cls'], 0),
            'target_img': np.concatenate(self.stats['target_img'], 0)
        }
        
        logger.info(f"Total predictions: {len(stats['conf'])}")
        logger.info(f"Total targets: {len(stats['target_cls'])}")
        
        # Compute AP per class
        results = ap_per_class(
            tp=stats['tp'],
            conf=stats['conf'],
            pred_cls=stats['pred_cls'],
            target_cls=stats['target_cls'],
            plot=False,
            names=self.class_names
        )
        
        # Unpack results
        tp, fp, p, r, f1, ap, unique_classes = results[:7]
        
        # Calculate mean metrics
        mp = p.mean() if len(p) > 0 else 0.0
        mr = r.mean() if len(r) > 0 else 0.0
        map50 = ap[:, 0].mean() if len(ap) > 0 else 0.0
        map50_95 = ap.mean() if len(ap) > 0 else 0.0
        
        # Per-class results
        per_class_results = {}
        for i, class_id in enumerate(unique_classes):
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            per_class_results[class_name] = {
                'precision': float(p[i]),
                'recall': float(r[i]),
                'f1': float(f1[i]),
                'ap50': float(ap[i, 0]),
                'ap50-95': float(ap[i].mean()),
                'tp': int(tp[i]),
                'fp': int(fp[i])
            }
        
        results_dict = {
            'precision': float(mp),
            'recall': float(mr),
            'mAP50': float(map50),
            'mAP50-95': float(map50_95),
            'per_class': per_class_results
        }
        
        return results_dict
    
    def print_results(self, results: Dict):
        """Print evaluation results in a formatted way"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"\nOverall Metrics:")
        print(f"  Precision:    {results['precision']:.4f}")
        print(f"  Recall:       {results['recall']:.4f}")
        print(f"  mAP@0.5:      {results['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {results['mAP50-95']:.4f}")
        
        print(f"\nPer-Class Results:")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<12} {'TP':<8} {'FP':<8}")
        print("-"*90)
        
        for class_name, metrics in results['per_class'].items():
            print(
                f"{class_name:<15} "
                f"{metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f} "
                f"{metrics['ap50']:<12.4f} "
                f"{metrics['ap50-95']:<12.4f} "
                f"{metrics['tp']:<8} "
                f"{metrics['fp']:<8}"
            )
        print("="*70)


def main():
    """Example usage"""
    
    # Define class mapping
    class_mapping = {
        'person': 1,
        'vehicle': 0,
        'bike': 2
    }
    
    # Paths to dataset
    images_path = "/path/to/images"
    labels_path = "/path/to/labels"
    
    # Create dataset
    dataset = YOLODataset(images_path, labels_path)
    
    # Create evaluator with visualization
    evaluator = SAM3Evaluator(
        class_mapping=class_mapping,
        device='cuda',
        conf_threshold=0.25,
        visualize=True,
        vis_output_dir='./visualizations',
        vis_mode='both'  # or 'predictions', 'ground_truth', 'comparison'
    )
    
    # Evaluate
    results = evaluator.evaluate_dataset(dataset, save_vis_every=1)
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
