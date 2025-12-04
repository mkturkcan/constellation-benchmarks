# Constellation Benchmarks

Benchmarking and evaluation framework for object detection models on the Constellation dataset.

## Overview

This repository provides tools for evaluating object detection models through standardized benchmarking protocols and cross-dataset evaluation. The framework supports YOLO-based architectures and enables performance assessment across different hardware platforms.

## Components

### Benchmarking

The `general_benchmark` module measures inference latency for YOLO models with platform-specific optimizations:

- TensorRT export for GPU acceleration
- NCNN export for embedded systems (e.g., Raspberry Pi)
- Statistical analysis across multiple iterations

### Evaluation

The `evaluation` module implements metrics computation and cross-model comparison:

- **metrics.py**: Core evaluation metrics (IoU, precision, recall, mAP)
- **yolo_eval.py**: YOLO model evaluation with class mapping support
- **zeroshot_eval.py**: Zero-shot detection evaluation
- **sam3_yolo_evaluator.py**: SAM3 and YOLO model comparison
- **moondream3_yolo_eval.py**: Moondream3 and YOLO model comparison

### Platform Support

- **dragonwing_litert**: TensorFlow Lite Runtime benchmarking for edge devices

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy

## Installation

Clone the repository:

```bash
git clone https://github.com/mkturkcan/constellation-benchmarks.git
cd constellation-benchmarks
```

Install dependencies:

```bash
pip install torch torchvision ultralytics opencv-python numpy pillow tqdm
```

For TensorRT support (GPU benchmarking):

```bash
pip install nvidia-tensorrt
```

For NCNN support (embedded systems):

```bash
pip install ncnn
```

## Getting Started

### Dataset Setup

Organize the Constellation dataset in YOLO format:

```
constellation_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Quick Start

Run a benchmark on pre-trained models:

```bash
python general_benchmark/run_benchmark.py \
    --platform tensorrt \
    --models constellation_yolov8n.pt constellation_yolov8x.pt \
    --image-size 832 \
    --iterations 100
```

## Usage

### Model Benchmarking

```bash
python general_benchmark/run_benchmark.py --platform tensorrt --models model1.pt model2.pt --iterations 100
```

### Model Evaluation

Evaluation scripts support custom class mappings for cross-dataset assessment. Refer to individual evaluation modules for specific usage patterns.

## License

See [LICENSE](LICENSE) for details.