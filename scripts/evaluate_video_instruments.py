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


def evaluate(sources, legacy, candidate):
    manifest = json.loads((ROOT/'reference/xr4-instrument-frames.json').read_text())['sources']
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
    output = {}
    for name, rows in samples.items():
        threshold = .40 if name == 'legacy' else max(
            (.1, .15, .2, .25, .3, .35, .4, .5, .6, .7),
            key=lambda threshold: (score(rows['6502766'], threshold)['f1'], threshold))
        heldout = [item for source in ('braces', '33337556') for item in rows[source]]
        output[name] = {'score_threshold': threshold, 'held_out': score(heldout, threshold),
            'by_source': {source: {'frames': len(items), 'split': manifest[source]['split'], **score(items, threshold)}
                          for source,items in rows.items()}}
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', type=Path)
    parser.add_argument('bedside_video', type=Path)
    parser.add_argument('legacy', type=Path)
    parser.add_argument('--candidate', type=Path)
    options = parser.parse_args()
    clips = capture_sources(options.sources)
    clips['5892118'] = options.bedside_video
    print(json.dumps(evaluate(clips, options.legacy, options.candidate), indent=2))
