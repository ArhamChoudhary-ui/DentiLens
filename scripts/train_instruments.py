"""Fine-tune the local instrument model and measure it on the prepared test split."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import shutil
from time import perf_counter

import torch
from ultralytics import YOLO


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('backbone', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--device', choices=('cpu','mps'), default='cpu')
    parser.add_argument('--install', type=Path)
    options = parser.parse_args()
    if options.output.exists() or not options.backbone.is_file():
        parser.error('Supply a local backbone and a new output folder.')
    if options.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is unavailable. Run with --device cpu.')
    torch.set_num_threads(min(6, torch.get_num_threads()))
    data = options.dataset.resolve() / 'data.yaml'
    audit = json.loads((data.parent / 'audit.json').read_text())
    model = YOLO(str(options.backbone.resolve()), task='obb')
    started = perf_counter()
    trained = model.train(data=str(data), imgsz=320, epochs=options.epochs,
                          batch=16, device=options.device, workers=0, seed=35,
                          project=str(options.output.resolve()), name='fit', optimizer='AdamW',
                          freeze=8, amp=False, mosaic=0, plots=False, cache=False,
                          fliplr=.3, flipud=0, degrees=8, shear=0, perspective=0)
    weight = Path(trained.save_dir) / 'weights' / 'best.pt'
    model = YOLO(str(weight), task='obb')
    evaluation = model.val(data=str(data), split='test', imgsz=320, device=options.device, workers=0,
                           plots=False, project=str(options.output.resolve()), name='test')
    per_class = {model.names[int(k)]: dict(precision=float(evaluation.box.p[i]),
                  recall=float(evaluation.box.r[i]), ap50=float(evaluation.box.ap50[i]),
                  ap50_95=float(evaluation.box.ap[i])) for i,k in enumerate(evaluation.box.ap_class_index)}
    result = dict(dataset_sha256=audit['archive_sha256'], classes=model.names,
                  weights_sha256=digest(weight), foundation_sha256=digest(options.backbone),
                  training_seconds=round(perf_counter()-started,2), epochs=options.epochs,
                  image_size=320, device=options.device, per_class=per_class,
                  test={name:float(value) for name,value in evaluation.results_dict.items()},
                  scope='Held-out still images after filename grouping and exact deduplication.')
    output = json.dumps(result, indent=2, allow_nan=False) + '\n'
    (options.output / 'evaluation.json').write_text(output)
    if options.install:
        options.install.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(weight, options.install / 'dental-instruments-obb.pt')
        (options.install / 'dental-instruments-obb.json').write_text(output)
    print(output, flush=True)


if __name__ == '__main__':
    main()
