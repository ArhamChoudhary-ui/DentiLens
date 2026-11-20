"""Evaluate a fixed detector on the held-out still-image test split once."""
import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new evaluation output directory')
    model = YOLO(str(args.weights))
    results = model.val(data=str(args.data.resolve()), split='test', imgsz=416,
                        batch=16, device='cpu', workers=0, plots=False,
                        conf=0.001, iou=0.7, project=str(args.output.resolve().parent),
                        name=args.output.name, exist_ok=False)
    report = {'weights_sha256': hashlib.file_digest(args.weights.open('rb'), 'sha256').hexdigest(),
              'split': 'test', 'images': len(results.box.image_metrics),
              'image_size': 416, 'device': 'cpu',
              'precision': float(results.box.mp), 'recall': float(results.box.mr),
              'mAP50': float(results.box.map50), 'mAP50_95': float(results.box.map),
              'metric_scope': 'One-class tooth detection boxes on supplied intraoral still images.',
              'precision_recall_scope': 'Ultralytics precision and recall at its F1-selected confidence operating point; not the app threshold.',
              'limitations': ['Does not measure video performance or SAM mask quality.',
                              'Patient-level independence of dataset splits is unknown.',
                              'Single training run with validation-based checkpoint selection.']}
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
