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


def measure_instruments(source, video, models, detector=None, *, experimental=False):
    started = perf_counter()
    device = 'cpu'
    with tempfile.TemporaryDirectory(prefix='tracker-', dir=source.parent) as temporary:
        config = Path(temporary) / 'instrument-bytetrack.yaml'
        config.write_text('tracker_type: bytetrack\ntrack_high_thresh: 0.40\ntrack_low_thresh: 0.25\n'
                          'new_track_thresh: 0.42\ntrack_buffer: 3\nmatch_thresh: 0.70\nfuse_score: true\n')
        if detector is None and not experimental:
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
        if experimental:
            import torch
            from ultralytics import YOLO

            weight = models / 'video-instruments.pt'
            report = json.loads((models.parent / 'reference/instrument-video-v3-results.json').read_text())
            with weight.open('rb') as stream:
                if file_digest(stream, 'sha256').hexdigest() != report['checkpoint_sha256']:
                    raise ValueError('Experimental checkpoint does not match its evaluation record.')
            torch.set_num_threads(min(4, torch.get_num_threads()))
            model = YOLO(str(weight), task='detect')
            threshold = report['comparison']['pilot']['score_threshold']

            def detector(frame):
                boxes = model.predict(frame, imgsz=report['image_size'], conf=threshold,
                                      device=device, verbose=False)[0].boxes
                return [('Dental instrument · type uncertain', score, [[x,y],[r,y],[r,b],[x,b]], None)
                        for (x,y,r,b), score in zip(boxes.xyxyn.tolist(), boxes.conf.tolist())]

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
                    observations.append({'detection_id': f'instrument-{"video-" if experimental else ""}{index}-{i}', 'track_id': f'I-{track}' if track is not None else None,
                                         'class': label, 'confidence': confidence, 'polygon': points,
                                         'box': [left, top, right, bottom], 'frame': index,
                                         'time': index / video['fps'], 'review_status': 'suggested'})
        finally:
            capture.release()
    return {'device': device, 'samples': samples, 'seconds': round(perf_counter() - started, 2),
            'observations': observations, 'appearances': group_instruments(observations, stride)}


def track_reviewed_region(source, video, frame_index, box):
    """Follow a reviewer-drawn region while bidirectional image matches agree."""
    import numpy as np

    capture = cv2.VideoCapture(str(source))
    fps = video['fps']

    def read(index):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        if not ok:
            raise ValueError(f'Could not read source frame {index}.')
        height, width = image.shape[:2]
        if width > 960:
            image = cv2.resize(image, (960, round(height * 960 / width)), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    try:
        anchor = read(frame_index)
        height, width = anchor.shape
        dimensions = np.array([width, height], dtype=np.float32)
        original = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]) * dimensions
        left, top = original[0].astype(int)
        right, bottom = np.ceil(original[2]).astype(int)
        mask = np.zeros_like(anchor)
        mask[top:bottom, left:right] = 255
        corners = cv2.goodFeaturesToTrack(anchor, 90, .015, 3, mask=mask)
        if corners is None or len(corners) < 6:
            return []
        samples = []
        step = max(1, round(fps / 8))
        for direction in (-1, 1):
            for offset in range(step, round(fps * 1.5) + 1, step):
                index = frame_index + direction * offset
                if index < 0 or index >= video['frames']:
                    break
                image = read(index)
                forward, forward_ok, _ = cv2.calcOpticalFlowPyrLK(anchor, image, corners, None)
                if forward is None:
                    break
                backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(image, anchor, forward, None)
                if backward is None:
                    break
                error = np.linalg.norm((backward - corners)[:, 0], axis=1)
                keep = (forward_ok[:, 0] > 0) & (backward_ok[:, 0] > 0) & (error < 1.5)
                if keep.sum() < max(6, len(corners) * .35):
                    break
                transform, inliers = cv2.estimateAffinePartial2D(
                    corners[keep], forward[keep], ransacReprojThreshold=2.5)
                if transform is None or inliers.sum() < max(6, keep.sum() * .65):
                    break
                scale = np.linalg.norm(transform[0, :2])
                if not .70 <= scale <= 1.35:
                    break
                moved = cv2.transform(original[None, :, :], transform)[0] / dimensions
                if np.any(moved < -.02) or np.any(moved > 1.02):
                    break
                bounds = np.clip(np.concatenate((moved.min(axis=0), moved.max(axis=0))), 0, 1)
                samples.append(dict(time=index / fps, frame=index, box=bounds.tolist()))
        return sorted(samples, key=lambda item: item['frame'])
    finally:
        capture.release()
