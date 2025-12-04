import sys
from zeroshot_eval import ZeroShotDetector

images_path="constellation/images/val"
labels_path="constellation/labels/val"

models_to_run = ["iSEE-Laboratory/llmdet_large", "IDEA-Research/grounding-dino-base", "rziga/mm_grounding_dino_large_all"]
confidence_thresholds = [0.4, 0.4, 0.4]

class_mapping = {
    'vehicle': 0,
    'person': 1,
}

# Readable class names (for display)
class_names = {i: name for name, i in class_mapping.items()}

# Create evaluator
for model_id in range(len(models_to_run)):
    evaluator = ZeroShotDetector(
        images_path=images_path,
        labels_path=labels_path,
        class_mapping=class_mapping,
        class_names=class_names,
        model_id= models_to_run[model_id],
        confidence_threshold = confidence_thresholds[model_id]
    )

    # Run evaluation
    metrics = evaluator.evaluate(verbose=True)

    # Print results
    evaluator.print_results(metrics)
