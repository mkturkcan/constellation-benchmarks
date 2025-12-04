"""
Simplified metrics module based on Ultralytics implementation
Contains essential functions for calculating precision, recall, and mAP
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate intersection-over-union (IoU) of boxes.
    
    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes in (x1, y1, x2, y2) format.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes in (x1, y1, x2, y2) format.
        eps (float, optional): A small value to avoid division by zero.
    
    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    
    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the average precision (AP) given the recall and precision curves.
    
    Args:
        recall: The recall curve.
        precision: The precision curve.
    
    Returns:
        ap (float): Average precision.
        mpre (np.ndarray): Precision envelope curve.
        mrec (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    
    # Integrate area under curve using 101-point interpolation (COCO)
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    
    return ap, mpre, mrec


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    on_plot=None,
    save_dir=None,
    names: Dict[int, str] = {},
    eps: float = 1e-16,
    prefix: str = "",
) -> Tuple:
    """
    Compute the average precision per class for object detection evaluation.
    
    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not.
        on_plot (callable, optional): A callback to pass plots path and data when they are rendered.
        save_dir: Directory to save the PR curves.
        names (dict[int, str], optional): Dictionary of class names to plot PR curves.
        eps (float, optional): A small value to avoid division by zero.
        prefix (str, optional): A prefix string for saving the plot files.
    
    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.
        fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class.
        p (np.ndarray): Precision values at threshold given by max F1 metric for each class.
        r (np.ndarray): Recall values at threshold given by max F1 metric for each class.
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class.
        ap (np.ndarray): Average precision for each class at different IoU thresholds.
        unique_classes (np.ndarray): An array of unique classes that have data.
        p_curve (np.ndarray): Precision curves for each class.
        r_curve (np.ndarray): Recall curves for each class.
        f1_curve (np.ndarray): F1-score curves for each class.
        x (np.ndarray): X-axis values for the curves.
        prec_values (np.ndarray): Precision values at mAP@0.5 for each class.
    """
    # Sort by objectness (confidence)
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections
    
    # Create Precision-Recall curve and compute AP for each class
    x = np.linspace(0, 1, 1000)
    prec_values = []
    
    # Initialize arrays
    ap = np.zeros((nc, tp.shape[1]))  # AP for each class at each IoU threshold
    p_curve = np.zeros((nc, 1000))  # Precision curve
    r_curve = np.zeros((nc, 1000))  # Recall curve
    
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels (ground truth) for this class
        n_p = i.sum()  # number of predictions for this class
        
        if n_p == 0 or n_l == 0:
            continue
        
        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        
        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        # Negative x (confidence), xp because xp decreases
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)
        
        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)
        
        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5
    
    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))
    
    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    
    # Find max F1 index
    def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
        """Box filter of fraction f."""
        nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
        p = np.ones(nf // 2)  # ones padding
        yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
        return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed
    
    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values


class Metric:
    """
    Class for computing evaluation metrics for YOLO models.
    
    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.
    """
    
    def __init__(self) -> None:
        """Initialize a Metric instance."""
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0
    
    @property
    def ap50(self) -> np.ndarray:
        """Return the Average Precision (AP) at an IoU threshold of 0.5 for all classes."""
        return self.all_ap[:, 0] if len(self.all_ap) else []
    
    @property
    def ap(self) -> np.ndarray:
        """Return the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes."""
        return self.all_ap.mean(1) if len(self.all_ap) else []
    
    @property
    def mp(self) -> float:
        """Return the Mean Precision of all classes."""
        return self.p.mean() if len(self.p) else 0.0
    
    @property
    def mr(self) -> float:
        """Return the Mean Recall of all classes."""
        return self.r.mean() if len(self.r) else 0.0
    
    @property
    def map50(self) -> float:
        """Return the mean Average Precision (mAP) at an IoU threshold of 0.5."""
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0
    
    @property
    def map75(self) -> float:
        """Return the mean Average Precision (mAP) at an IoU threshold of 0.75."""
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0
    
    @property
    def map(self) -> float:
        """Return the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95."""
        return self.all_ap.mean() if len(self.all_ap) else 0.0
    
    def mean_results(self):
        """Return mean of results: mp, mr, map50, map."""
        return [self.mp, self.mr, self.map50, self.map]
    
    def class_result(self, i: int):
        """Return class-aware result: p[i], r[i], ap50[i], ap[i]."""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]
    
    @property
    def maps(self) -> np.ndarray:
        """Return mAP of each class."""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps
    
    def update(self, results: Tuple):
        """
        Update the evaluation metrics with a new set of results.
        
        Args:
            results: Tuple containing (p, r, f1, all_ap, ap_class_index, ...)
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
        ) = results[:5]
