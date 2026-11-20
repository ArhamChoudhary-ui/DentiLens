"""Fine-tune a compact one-class detector, retaining validation-only selection."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('pretrained', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        parser.error('Choose a new training output directory')
    backbone = json.loads((Path(__file__).resolve().parents[1] / 'reference/tooth-backbone.json').read_text())
    with args.pretrained.open('rb') as stream:
        pretrained_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if pretrained_digest != backbone['sha256']:
        parser.error('Use the pinned backbone from scripts/download_tooth_backbone.py')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / 'training-process.json').write_text(json.dumps({'pid': os.getpid()}))
    model = YOLO(str(args.output / 'weights/last.pt') if args.resume else str(args.pretrained))
    def set_threads(trainer):
        import torch
        torch.set_num_threads(args.threads)
    model.add_callback('on_train_start', set_threads)
    if args.resume:
        model.train(resume=True, device=args.device, workers=0)
    else:
        model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=0, cache=False,
                project=str(args.output.resolve().parent), name=args.output.name,
                seed=42, deterministic=True, optimizer='AdamW', lr0=0.001,
                freeze=10, patience=5, close_mosaic=3, mosaic=0.5,
                fliplr=0.0, flipud=0.0, degrees=5.0, scale=0.2,
                    plots=False, amp=False, save=True, exist_ok=False)
    weights = args.output / 'weights/best.pt'
    rows = list(csv.DictReader((args.output / 'results.csv').open()))
    provenance = {'name': 'DentiLens intraoral YOLOv8n tooth detector',
                  'weights_sha256': hashlib.file_digest(weights.open('rb'), 'sha256').hexdigest(),
                  'pretrained_sha256': pretrained_digest, 'pretrained_source': backbone['url'],
                  'dataset': json.loads((args.data.parent / 'preparation.json').read_text()),
                  'training': {'epochs_requested': args.epochs, 'epochs_completed': len(rows),
                               'resumed_from_checkpoint': args.resume, 'cpu_threads': args.threads,
                               'image_size': args.imgsz,
                               'batch': args.batch, 'device': args.device, 'frozen_layers': 10,
                               'seed': 42, 'selection': 'best validation fitness; test split unused'},
                  'limitations': ['Single tooth class; no FDI numbering.',
                                  'Detection metrics do not measure SAM segmentation accuracy.',
                                  'Intraoral still-image performance does not establish video accuracy.']}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Best detector: {weights}')


if __name__ == '__main__':
    main()
