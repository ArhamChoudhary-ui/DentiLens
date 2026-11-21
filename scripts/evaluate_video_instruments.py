"""Compare instrument localization on the reviewed frame and glove-crop manifest."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from train_video_instruments import ROOT, capture_sources


def iou(a, b):
    left, top = np.maximum(a[:2], b[:2])
    right, bottom = np.minimum(a[2:], b[2:])
    intersection = max(0, right-left) * max(0, bottom-top)
    return intersection / ((a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - intersection)


def score(samples, confidence, overlap=.5):
    tp = fp = fn = 0
    for expected, predicted in samples:
        remaining = list(expected)
        for weight, box in sorted(predicted, key=lambda row: -row[0]):
            if weight < confidence:
                continue
            matches = [iou(box, other) for other in remaining]
            if matches and max(matches) >= overlap:
                tp += 1
                remaining.pop(int(np.argmax(matches)))
            else:
                fp += 1
        fn += len(remaining)
    return dict(tp=tp, fp=fp, fn=fn, precision=round(tp/max(1,tp+fp), 4),
                recall=round(tp/max(1,tp+fn), 4), f1=round(2*tp/max(1,2*tp+fn+fp), 4))


def predict(model, frame, size):
    result = model.predict(frame, imgsz=size, conf=.01, device='cpu', verbose=False)[0]
    if result.obb is not None:
        obb = result.obb
        points = obb.xyxyxyxyn.cpu().numpy()
        return list(zip(obb.conf.cpu().tolist(), np.concatenate((points.min(axis=1), points.max(axis=1)), axis=1)))
    return list(zip(result.boxes.conf.cpu().tolist(), result.boxes.xyxyn.cpu().numpy()))


def evaluate(sources, legacy, candidate, reflection=None, portrait=None, portrait_split=False):
    manifest = json.loads((ROOT/'reference/xr4-instrument-frames.json').read_text())['sources']
    if portrait_split:
        manifest['6502766']['split'] = 'related'
        manifest['33337556']['split'] = 'valid'
    validation = [source for source, group in manifest.items() if group['split'] == 'valid']
    testing = [source for source, group in manifest.items() if group['split'] == 'test']
    torch.set_num_threads(4)
    models = {'legacy': (YOLO(str(legacy)), 320)}
    if candidate:
        models['pilot'] = (YOLO(str(candidate)), 512)
    samples = {key: {} for key in models}
    grid = np.array([500, 280, 500, 280])
    for source, group in manifest.items():
        cap = cv2.VideoCapture(str(sources[source]))
        try:
            for number, bounds in (group.get('frames') or group['negative_crops']).items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(number))
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Could not decode {source} frame {number}.')
                expected = [box / grid for box in bounds] if 'frames' in group else []
                if 'negative_crops' in group:
                    h, w = frame.shape[:2]
                    left, top, right, bottom = np.round(bounds / grid * [w, h, w, h]).astype(int)
                    frame = frame[top:bottom, left:right]
                for name,(model,size) in models.items():
                    samples[name].setdefault(source, []).append((expected, predict(model, frame, size)))
        finally:
            cap.release()
    if portrait:
        training = json.loads((ROOT/'reference/instrument-video-v3-frames.json').read_text())
        cap = cv2.VideoCapture(str(portrait))
        try:
            frame_grid = np.tile(training['grid'], 2)
            for number, boxes in training['frames'].items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(number))
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Could not decode portrait frame {number}.')
                expected = [box / frame_grid for box in boxes]
                for name, (model, size) in models.items():
                    samples[name].setdefault('6502638', []).append((expected, predict(model, frame, size)))
        finally:
            cap.release()
    if reflection:
        cap = cv2.VideoCapture(str(reflection))
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.linspace(0, frame_count - 1, min(30, frame_count), dtype=int)
            for frame_index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Could not decode reflection frame {frame_index}.')
                for name, (model, size) in models.items():
                    samples[name].setdefault('6529138', []).append(([], predict(model, frame, size)))
        finally:
            cap.release()
    output = {}
    for name, rows in samples.items():
        validation_rows = [item for source in validation for item in rows[source]]
        choices = (.1, .15, .2, .25, .3, .35, .4, .5, .6, .7)
        at_validation = {cutoff: score(validation_rows, cutoff) for cutoff in choices}
        threshold = .40 if name == 'legacy' else max(
            choices, key=lambda cutoff: (at_validation[cutoff]['f1'], cutoff))
        heldout = [item for source in testing for item in rows[source]]
        output[name] = {'score_threshold': threshold, 'validation': at_validation[threshold],
            'held_out': score(heldout, threshold),
            'by_source': {source: {'frames': len(items), 'split': manifest[source]['split'], **score(items, threshold)}
                          for source,items in rows.items() if source in manifest}}
        if portrait_split and name == 'pilot':
            precise = [cutoff for cutoff, result in at_validation.items()
                       if result['tp'] >= 3 and result['precision'] >= .8]
            if precise:
                verified = max(precise, key=lambda cutoff: (at_validation[cutoff]['recall'], cutoff))
                output[name]['precision_operating_point'] = {
                    'score_threshold': verified, 'validation': at_validation[verified],
                    'held_out': score(heldout, verified),
                    'reflection_boxes': score(rows['6529138'], verified)['fp'] if reflection else None,
                }
            else:
                output[name]['precision_operating_point'] = None
        if portrait:
            output[name]['portrait_training_check'] = {
                'source': '6502638', 'frames': len(rows['6502638']), **score(rows['6502638'], threshold),
            }
        if reflection:
            rows_reflection = rows['6529138']
            output[name]['reflection_check'] = {
                'source': '6529138', 'sampled_frames': len(rows_reflection),
                'boxes_on_reflection_clip': score(rows_reflection, threshold)['fp'],
            }
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', type=Path)
    parser.add_argument('bedside_video', type=Path)
    parser.add_argument('legacy', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--reflection-video', type=Path)
    parser.add_argument('--portrait-video', type=Path)
    parser.add_argument('--portrait-split', action='store_true')
    options = parser.parse_args()
    clips = capture_sources(options.sources)
    clips['5892118'] = options.bedside_video
    print(json.dumps(evaluate(clips, options.legacy, options.candidate, options.reflection_video, options.portrait_video, options.portrait_split), indent=2))
