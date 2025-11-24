import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
from PIL import Image
import os, time, sys
import glob
from pathlib import Path

def curr_ms():
    return round(time.time() * 1000)

# Paths
MODELS_DIR = 'models'  # Directory containing .tflite models
IMAGES_DIR = 'images'  # Directory containing images
MAX_IMAGES = 100       # Maximum number of images to test

# Parse command line arguments
use_qnn = '--use-qnn' in sys.argv

# Load an image (using Pillow) and make it in the right format that the interpreter expects
def load_image_litert(interpreter, path, single_channel_behavior: str = 'grayscale'):
    d = interpreter.get_input_details()[0]
    shape = [int(x) for x in d["shape"]]  # e.g. [1, H, W, C] or [1, C, H, W]
    dtype = d["dtype"]
    scale, zp = d.get("quantization", (0.0, 0))

    if len(shape) != 4 or shape[0] != 1:
        raise ValueError(f"Unexpected input shape: {shape}")

    # Detect layout
    if shape[1] in (1, 3):   # [1, C, H, W]
        layout, C, H, W = "NCHW", shape[1], shape[2], shape[3]
    elif shape[3] in (1, 3): # [1, H, W, C]
        layout, C, H, W = "NHWC", shape[3], shape[1], shape[2]
    else:
        raise ValueError(f"Cannot infer layout from shape {shape}")

    # Load & resize
    img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    arr = np.array(img)
    if C == 1:
        if single_channel_behavior == 'grayscale':
            # Convert to luminance (H, W)
            gray = np.asarray(Image.fromarray(arr).convert('L'))
        elif single_channel_behavior in ('red', 'green', 'blue'):
            ch_idx = {'red': 0, 'green': 1, 'blue': 2}[single_channel_behavior]
            gray = arr[:, :, ch_idx]
        else:
            raise ValueError(f"Invalid single_channel_behavior: {single_channel_behavior}")
        # Keep shape as HWC with C=1
        arr = gray[..., np.newaxis]

    # HWC -> correct layout
    if layout == "NCHW":
        arr = np.transpose(arr, (2, 0, 1))  # (C,H,W)

    # Scale 0..1 (all AI Hub image models use this)
    arr = (arr / 255.0).astype(np.float32)

    # Quantize if needed
    if scale and float(scale) != 0.0:
        q = np.rint(arr / float(scale) + int(zp))
        if dtype == np.uint8:
            arr = np.clip(q, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(q, -128, 127).astype(np.int8)

    return np.expand_dims(arr, 0)  # add batch


def benchmark_model(model_path, image_tensors, use_qnn=False):
    """
    Benchmark a single model on pre-loaded images.
    
    Args:
        model_path: Path to the .tflite model
        image_tensors: List of preprocessed image tensors
        use_qnn: Whether to use QNN delegate
    
    Returns:
        dict with model_name, mean_ms, std_ms, num_images
    """
    print(f"\nBenchmarking: {os.path.basename(model_path)}")
    
    # Setup delegates
    experimental_delegates = []
    if use_qnn:
        experimental_delegates = [load_delegate("libQnnTFLiteDelegate.so", 
                                               options={"backend_type":"htp"})]
    
    # Load model
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=experimental_delegates
        )
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"  Error loading model: {e}")
        return None
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Warmup run
    interpreter.set_tensor(input_details[0]['index'], image_tensors[0])
    interpreter.invoke()
    
    # Benchmark inference times
    inference_times = []
    
    for img_tensor in image_tensors:
        interpreter.set_tensor(input_details[0]['index'], img_tensor)
        
        start = curr_ms()
        interpreter.invoke()
        end = curr_ms()
        
        inference_times.append(end - start)
    
    # Calculate statistics
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    
    print(f"  Processed {len(image_tensors)} images")
    print(f"  Mean: {mean_time:.2f} ms, Std: {std_time:.2f} ms")
    
    return {
        'model_name': os.path.basename(model_path),
        'model_path': model_path,
        'mean_ms': mean_time,
        'std_ms': std_time,
        'num_images': len(image_tensors)
    }


def main():
    print("="*70)
    print("Model Benchmarking Tool")
    print("="*70)
    print(f"Models directory: {MODELS_DIR}")
    print(f"Images directory: {IMAGES_DIR}")
    print(f"Max images: {MAX_IMAGES}")
    print(f"Using QNN: {use_qnn}")
    print("="*70)
    
    # Find all .tflite models
    model_paths = sorted(glob.glob(os.path.join(MODELS_DIR, "*.tflite")))
    
    if not model_paths:
        print(f"\nError: No .tflite models found in {MODELS_DIR}")
        return
    
    print(f"\nFound {len(model_paths)} model(s):")
    for mp in model_paths:
        print(f"  - {os.path.basename(mp)}")
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
        image_paths.extend(glob.glob(os.path.join(IMAGES_DIR, ext.upper())))
    
    image_paths = sorted(set(image_paths))[:MAX_IMAGES]
    
    if not image_paths:
        print(f"\nError: No images found in {IMAGES_DIR}")
        return
    
    print(f"\nFound {len(image_paths)} image(s) to process")
    
    # Results storage
    results = []
    
    # Benchmark each model
    for model_path in model_paths:
        # Load model to get input requirements
        try:
            experimental_delegates = []
            if use_qnn:
                experimental_delegates = [load_delegate("libQnnTFLiteDelegate.so", 
                                                       options={"backend_type":"htp"})]
            
            temp_interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=experimental_delegates
            )
            temp_interpreter.allocate_tensors()
            
            # Pre-load and preprocess all images
            print(f"\n  Preprocessing {len(image_paths)} images...")
            image_tensors = []
            
            # Determine single channel behavior based on model
            # You may need to adjust this logic based on your models
            single_channel_behavior = 'blue'  # Default for face detection models
            
            for i, img_path in enumerate(image_paths):
                try:
                    tensor = load_image_litert(temp_interpreter, img_path, 
                                              single_channel_behavior=single_channel_behavior)
                    image_tensors.append(tensor)
                    
                    if (i + 1) % 20 == 0:
                        print(f"    Preprocessed {i + 1}/{len(image_paths)} images...")
                except Exception as e:
                    print(f"    Warning: Failed to load {os.path.basename(img_path)}: {e}")
            
            if not image_tensors:
                print(f"  Error: No images could be preprocessed for this model")
                continue
            
            # Now benchmark with pre-loaded images
            result = benchmark_model(model_path, image_tensors, use_qnn)
            
            if result:
                results.append(result)
                
        except Exception as e:
            print(f"  Error processing model: {e}")
            continue
    
    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    
    if not results:
        print("No results to display")
        return
    
    # Sort by mean time
    results.sort(key=lambda x: x['mean_ms'])
    
    print(f"\n{'Model':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'Images':<8}")
    print("-"*70)
    
    for result in results:
        print(f"{result['model_name']:<30} "
              f"{result['mean_ms']:>10.2f}  "
              f"{result['std_ms']:>10.2f}  "
              f"{result['num_images']:>6}")
    
    print("="*70)
    
    # Save results to CSV
    csv_path = 'benchmark_results.csv'
    with open(csv_path, 'w') as f:
        f.write("model_name,mean_ms,std_ms,num_images,model_path\n")
        for result in results:
            f.write(f"{result['model_name']},{result['mean_ms']:.4f},"
                   f"{result['std_ms']:.4f},{result['num_images']},"
                   f"{result['model_path']}\n")
    
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
