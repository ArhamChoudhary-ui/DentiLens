"""Refine the generic instrument-localization pilot using one reviewed portrait clip."""
import argparse
import json
from pathlib import Path
import random
from zipfile import ZipFile

import cv2
import torch
import yaml
from ultralytics import YOLO

from train_video_instruments import ROOT


def add_photo_examples(archive_path, target, per_class=120):
    # Select one export per original and use only the supplied training split.
    with ZipFile(archive_path) as archive:
        classes = yaml.safe_load(archive.read('data.yaml'))['names']
        target_classes = {'Dental Mirrors', 'Mouth mirror', 'Explorer No.5'}
        mapped = {int(key) for key, label in classes.items() if label in target_classes}
        if len(mapped) != len(target_classes):
            raise ValueError('The archive does not contain the expected mirror and explorer labels.')
        explorers = {int(key) for key, label in classes.items() if label == 'Explorer No.5'}
        selected = {'mirror': {}, 'explorer': {}}
        for name in sorted(archive.namelist()):
            if not name.startswith('train/labels/') or not name.endswith('.txt'):
                continue
            stem = Path(name).stem
            group = stem.split('.rf.', 1)[0]
            regions = [line.split() for line in archive.read(name).decode().splitlines() if line.strip()]
            labels = {int(region[0]) for region in regions}
            if labels & explorers:
                selected['explorer'].setdefault(group, (stem, regions))
            if labels & (mapped - explorers):
                selected['mirror'].setdefault(group, (stem, regions))

        samples = {}
        rng = random.Random(44)
        for groups in selected.values():
            keys = rng.sample(sorted(groups), min(per_class, len(groups)))
            samples.update((key, groups[key]) for key in keys)
        for stem, regions in samples.values():
            lines = []
            for label, *values in regions:
                # The localized category is a handheld instrument. Trays and other
                # supplies are retained in the image as context.
                if classes[int(label)] in ('Galipot', 'Kidney Dish', 'Mouth Props'):
                    continue
                if len(values) != 8:
                    raise ValueError(f'Invalid oriented annotation in {stem}.')
                coordinates = [float(value) for value in values]
                if not all(0 <= value <= 1 for value in coordinates):
                    raise ValueError(f'Out-of-frame annotation in {stem}.')
                left, right = min(coordinates[::2]), max(coordinates[::2])
                top, bottom = min(coordinates[1::2]), max(coordinates[1::2])
                if left < right and top < bottom:
                    lines.append(f'0 {(left+right)/2:.7f} {(top+bottom)/2:.7f} {right-left:.7f} {bottom-top:.7f}')
            if not lines:
                continue
            key = 'v3-' + stem
            (target / 'images' / f'{key}.jpg').write_bytes(archive.read(f'train/images/{stem}.jpg'))
            (target / 'labels' / f'{key}.txt').write_text('\n'.join(lines) + '\n')
        return len(samples)


def prepare(old_data, portrait, destination, archive=None):
    for split in ('train', 'valid', 'test'):
        for kind in ('images', 'labels'):
            (destination / split / kind).mkdir(parents=True)
    for kind in ('images', 'labels'):
        for file in (old_data / 'train' / kind).iterdir():
            if file.is_file():
                (destination / 'train' / kind / file.name).symlink_to(file.resolve())
        for file in (old_data / 'test' / kind).iterdir():
            if file.name.startswith('33337556-'):
                split = 'valid'
            elif file.name.startswith('braces-'):
                split = 'test'
            else:
                continue
            (destination / split / kind / file.name).symlink_to(file.resolve())

    manifest = json.loads((ROOT / 'reference/instrument-video-v3-frames.json').read_text())
    grid_w, grid_h = manifest['grid']
    capture = cv2.VideoCapture(str(portrait))
    try:
        for index, boxes in manifest['frames'].items():
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Could not decode frame {index}.')
            # Keep both the full-frame geometry and a closer crop with independent context.
            for width, height in [(grid_w, grid_h), (grid_w, 280)]:
                h, w = frame.shape[:2]
                crop = frame[:round(height * h / grid_h), :round(width * w / grid_w)]
                key = f'{manifest["source"]}-{index}-{height}'
                output = cv2.resize(crop, (width * 2, height * 2), interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(destination / 'train/images' / f'{key}.jpg'), output):
                    raise OSError(f'Could not write frame {key}.')
                labels = [f'0 {(x1+x2)/2/width:.7f} {(y1+y2)/2/height:.7f} '
                          f'{(x2-x1)/width:.7f} {(y2-y1)/height:.7f}' for x1,y1,x2,y2 in boxes]
                (destination / 'train/labels' / f'{key}.txt').write_text('\n'.join(labels) + '\n')
    finally:
        capture.release()
    if archive:
        count = add_photo_examples(archive, destination / 'train')
        print(f'Selected {count} original photo groups from the V3 export.')
    config = destination / 'data.yaml'
    config.write_text(yaml.safe_dump(dict(path=str(destination.resolve()), train='train/images',
                                          val='valid/images', test='test/images', names={0:'Dental instrument'})))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous_data', type=Path)
    parser.add_argument('portrait', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    options = parser.parse_args()
    options.output.mkdir(parents=True)
    data = prepare(options.previous_data, options.portrait, options.output / 'data', options.archive)
    torch.set_num_threads(6)
    model = YOLO(str(options.checkpoint.resolve()))
    result = model.train(data=str(data.resolve()), imgsz=512, epochs=options.epochs, batch=12,
                         workers=0, device=options.device, project=str(options.output.resolve()), name='fit',
                         optimizer='AdamW', lr0=.0005, freeze=6, warmup_epochs=0,
                         amp=False, plots=False, seed=44, mosaic=0, flipud=0, degrees=8, hsv_s=.35)
    best = Path(result.save_dir) / 'weights/best.pt'
    scores = YOLO(str(best)).val(data=str(data.resolve()), split='test', imgsz=512, device='cpu',
                                workers=0, plots=False, project=str(options.output.resolve()), name='test')
    (options.output / 'evaluation.json').write_text(json.dumps(scores.results_dict, indent=2) + '\n')


if __name__ == '__main__':
    main()
