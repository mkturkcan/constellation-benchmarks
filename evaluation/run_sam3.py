from sam3_yolo_evaluator_with_vis import SAM3Evaluator, YOLODataset

dataset = YOLODataset("constellation/images/val/", "constellation/labels/val/")
evaluator = SAM3Evaluator(
    class_mapping={'person': 1, 'vehicle': 0},
    device='cuda',
    conf_threshold=0.7,
    visualize=True,
    vis_output_dir='./visualizations',
    vis_mode='both' 
)

# Evaluate
results = evaluator.evaluate_dataset(dataset, save_vis_every=10)

evaluator.print_results(results)

import json
with open('evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
