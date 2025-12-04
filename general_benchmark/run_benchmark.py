#!/usr/bin/env python3
"""
YOLO Model Benchmarking Script

Benchmark YOLO models with platform-specific optimizations (TensorRT or NCNN).
Measures inference time across multiple iterations for statistical analysis.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# Configuration Constants
DEFAULT_IMAGE_SIZE = 832
DEFAULT_NUM_ITERATIONS = 100
DEFAULT_WARMUP_ITERATIONS = 1
TRAIN_IMAGE_DIR = Path("constellation_dataset/images/train")

# Platform-specific export settings
EXPORT_CONFIG = {
    "tensorrt": {"format": "engine", "extension": ".engine"},
    "ncnn": {"format": "ncnn", "extension": "_ncnn_model"},
}


class ModelBenchmark:
    """Benchmark YOLO models with specified export format."""

    def __init__(
        self,
        model_paths: List[str],
        export_format: str = "tensorrt",
        image_size: int = DEFAULT_IMAGE_SIZE,
        num_iterations: int = DEFAULT_NUM_ITERATIONS,
    ):
        """
        Initialize the benchmark configuration.

        Args:
            model_paths: List of model file paths to benchmark
            export_format: Export format ('tensorrt' or 'ncnn')
            image_size: Input image size for inference
            num_iterations: Number of iterations for timing measurements
        """
        self.model_paths = model_paths
        self.export_format = export_format
        self.image_size = image_size
        self.num_iterations = num_iterations
        self.export_config = EXPORT_CONFIG[export_format]

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def get_exported_model_path(self, model_path: Path) -> Path:
        """
        Generate the exported model path based on the export format.

        Args:
            model_path: Original model path

        Returns:
            Path to the exported model
        """
        extension = self.export_config["extension"]
        return model_path.parent / model_path.name.replace(".pt", extension)

    def load_or_export_model(self, model_path: Path) -> YOLO:
        """
        Load an exported model or export it if not available.

        Args:
            model_path: Path to the original model

        Returns:
            Loaded YOLO model instance

        Raises:
            FileNotFoundError: If the original model file doesn't exist
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        exported_path = self.get_exported_model_path(model_path)

        if exported_path.exists():
            self.logger.info(f"Loading existing exported model: {exported_path}")
            return YOLO(str(exported_path))

        self.logger.info(f"Exporting model to {self.export_format} format...")
        model = YOLO(str(model_path))
        model.export(
            format=self.export_config["format"],
            simplify=True,
            half=True,
        )
        self.logger.info(f"Export complete: {exported_path}")

        return YOLO(str(exported_path))

    def warmup_model(self, model: YOLO, warmup_image_path: Path) -> None:
        """
        Perform warmup inference to initialize the model.

        Args:
            model: YOLO model instance
            warmup_image_path: Path to warmup image
        """
        if not warmup_image_path.exists():
            self.logger.warning(f"Warmup image not found: {warmup_image_path}")
            return

        self.logger.info("Performing warmup inference...")
        for _ in range(DEFAULT_WARMUP_ITERATIONS):
            model.predict(
                str(warmup_image_path),
                imgsz=self.image_size,
                verbose=False,
            )

    def benchmark_model(self, model: YOLO) -> Tuple[float, float]:
        """
        Benchmark model inference time over multiple iterations.

        Args:
            model: YOLO model instance

        Returns:
            Tuple of (mean_time_ms, std_time_ms)

        Raises:
            ValueError: If no valid images are found for benchmarking
        """
        timings = []
        successful_iterations = 0

        for i in range(self.num_iterations):
            image_path = TRAIN_IMAGE_DIR / f"{i}.jpg"

            if not image_path.exists():
                self.logger.warning(f"Image not found: {image_path}, skipping...")
                continue

            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    self.logger.warning(f"Failed to read image: {image_path}")
                    continue

                image = np.ascontiguousarray(image)

                start_time = time.perf_counter()
                model.predict(image, imgsz=self.image_size, verbose=False)
                end_time = time.perf_counter()

                timings.append(end_time - start_time)
                successful_iterations += 1

            except Exception as e:
                self.logger.error(f"Error processing image {image_path}: {e}")
                continue

        if not timings:
            raise ValueError("No successful inference iterations completed")

        self.logger.info(
            f"Completed {successful_iterations}/{self.num_iterations} iterations"
        )

        mean_time_ms = np.mean(timings) * 1000
        std_time_ms = np.std(timings) * 1000

        return mean_time_ms, std_time_ms

    def run_benchmark(self) -> None:
        """Execute the benchmark for all specified models."""
        self.logger.info(
            f"Starting benchmark with {self.export_format} export format"
        )
        self.logger.info(f"Image size: {self.image_size}")
        self.logger.info(f"Iterations: {self.num_iterations}")
        self.logger.info("-" * 80)

        results = []

        for model_path_str in self.model_paths:
            model_path = Path(model_path_str)
            self.logger.info(f"\nBenchmarking: {model_path.name}")

            try:
                # Load or export model
                model = self.load_or_export_model(model_path)

                # Warmup
                warmup_image = TRAIN_IMAGE_DIR / "0.jpg"
                self.warmup_model(model, warmup_image)

                # Benchmark
                mean_time, std_time = self.benchmark_model(model)

                result = {
                    "model": model_path.name,
                    "mean_ms": mean_time,
                    "std_ms": std_time,
                }
                results.append(result)

                self.logger.info(
                    f"Results: {mean_time:.2f} ± {std_time:.2f} ms"
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to benchmark {model_path.name}: {e}"
                )
                continue

        # Print summary
        self.print_summary(results)

    def print_summary(self, results: List[dict]) -> None:
        """
        Print a formatted summary of benchmark results.

        Args:
            results: List of result dictionaries
        """
        if not results:
            self.logger.warning("No benchmark results to display")
            return

        self.logger.info("\n" + "=" * 80)
        self.logger.info("BENCHMARK SUMMARY")
        self.logger.info("=" * 80)

        for result in results:
            self.logger.info(
                f"{result['model']:<40} {result['mean_ms']:>8.2f} ± "
                f"{result['std_ms']:>6.2f} ms"
            )

        self.logger.info("=" * 80)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO models with platform-specific optimizations"
    )

    parser.add_argument(
        "--platform",
        type=str,
        choices=["tensorrt", "ncnn"],
        default="tensorrt",
        help="Target platform (default: tensorrt for GPU, ncnn for Raspberry Pi)",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "constellation_yolov8n.pt",
            "constellation_yolov8x.pt",
            "constellation_yolov8xp2p6.pt",
            "constellation_rtdetrx.pt",
        ],
        help="List of model files to benchmark",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Input image size (default: {DEFAULT_IMAGE_SIZE})",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_NUM_ITERATIONS,
        help=f"Number of benchmark iterations (default: {DEFAULT_NUM_ITERATIONS})",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the benchmark script."""
    args = parse_arguments()

    benchmark = ModelBenchmark(
        model_paths=args.models,
        export_format=args.platform,
        image_size=args.image_size,
        num_iterations=args.iterations,
    )

    benchmark.run_benchmark()


if __name__ == "__main__":
    main()