"""Sample the normalized recording with a local oriented instrument detector."""
from collections import defaultdict
from hashlib import file_digest
import json
import math
from pathlib import Path
import tempfile
from time import perf_counter

import cv2


def instrument_model_ready(models):
    return all((models / name).is_file() for name in ('dental-instruments-obb.pt', 'dental-instruments-obb.json'))


def display_name(raw):
    return {
        'airotor': 'Air turbine handpiece', 'forcep': 'Forceps', 'Mouth mirror': 'Dental mirror',
        'prophy': 'Prophy instrument', 'spoon excavators': 'Spoon excavator',
        'triple syringe dental': 'Three-way dental syringe',
    }.get(raw, raw[:1].upper() + raw[1:])


def group_instruments(observations, stride):
    tracks = defaultdict(list)
    for row in observations:
        key = row['track_id'] or row['detection_id']
        spans = tracks[(key, row['class'])]
        if spans and row['frame'] - spans[-1]['last_frame'] == stride:
            span = spans[-1]
            span.update(last_frame=row['frame'], end=row['time'], samples=span['samples'] + 1)
            if row['confidence'] > span['best']['confidence']:
                span['best'] = row
        else:
            spans.append(dict(id=row['detection_id'], track_id=row['track_id'], label=row['class'], time=row['time'],
                              end=row['time'], last_frame=row['frame'], samples=1, best=row))
    return sorted((span for spans in tracks.values() for span in spans), key=lambda span: span['time'])


def measure_instruments(source, video, models, detector=None):
    started = perf_counter()
    device = 'cpu'
    with tempfile.TemporaryDirectory(prefix='tracker-', dir=source.parent) as temporary:
        config = Path(temporary) / 'instrument-bytetrack.yaml'
        config.write_text('tracker_type: bytetrack\ntrack_high_thresh: 0.40\ntrack_low_thresh: 0.25\n'
                          'new_track_thresh: 0.42\ntrack_buffer: 3\nmatch_thresh: 0.70\nfuse_score: true\n')
        if detector is None:
            import torch
            from ultralytics import YOLO

            weight = models / 'dental-instruments-obb.pt'
            metadata = json.loads(weight.with_suffix('.json').read_text())
            with weight.open('rb') as stream:
                if file_digest(stream, 'sha256').hexdigest() != metadata['weights_sha256']:
                    raise ValueError('Instrument checkpoint does not match its evaluation record.')
            model = YOLO(str(weight), task='obb')
            if model.names != {int(k):v for k,v in metadata['classes'].items()}:
                raise ValueError('Instrument checkpoint classes do not match metadata.')
            # The model was evaluated and qualified in CPU mode.
            torch.set_num_threads(min(4, torch.get_num_threads()))

            def detector(frame):
                results = model.track(frame, task='obb', tracker=str(config), persist=True, conf=.40,
                                      imgsz=metadata['image_size'], verbose=False, device=device)
                obb = results[0].obb
                if obb is None or len(obb) == 0:
                    return []
                ids = obb.id.int().cpu().tolist() if obb.id is not None else [None] * len(obb)
                return [(display_name(model.names[label]), score, points, key)
                        for label,score,points,key in zip(obb.cls.int().cpu().tolist(), obb.conf.cpu().tolist(),
                                                        obb.xyxyxyxyn.cpu().tolist(), ids)]
        stride = max(1, round(video['fps'] / 5))
        capture = cv2.VideoCapture(str(source))
        observations = []
        samples = 0
        try:
            for index in range(video['frames']):
                if not capture.grab():
                    raise ValueError(f'Instrument scan stopped at frame {index}.')
                if index % stride:
                    continue
                ok, frame = capture.retrieve()
                if not ok:
                    raise ValueError(f'Instrument scan could not decode frame {index}.')
                samples += 1
                for i, (label, confidence, corners, track) in enumerate(detector(frame)):
                    if len(corners) != 4 or any(len(p) != 2 for p in corners):
                        continue
                    if not math.isfinite(confidence) or not all(math.isfinite(v) for p in corners for v in p):
                        continue
                    points = [[min(1, max(0, value)) for value in point] for point in corners]
                    left, right = sorted((min(p[0] for p in points), max(p[0] for p in points)))
                    top, bottom = sorted((min(p[1] for p in points), max(p[1] for p in points)))
                    if right <= left or bottom <= top:
                        continue
                    observations.append({'detection_id': f'instrument-{index}-{i}', 'track_id': f'I-{track}' if track is not None else None,
                                         'class': label, 'confidence': confidence, 'polygon': points,
                                         'box': [left, top, right, bottom], 'frame': index,
                                         'time': index / video['fps'], 'review_status': 'suggested'})
        finally:
            capture.release()
    return {'device': device, 'samples': samples, 'seconds': round(perf_counter() - started, 2),
            'observations': observations, 'appearances': group_instruments(observations, stride)}
