"""Associate detection anchors with ByteTrack and bridge frames with checked optical flow."""
from collections import Counter, deque
from dataclasses import dataclass, field
from types import SimpleNamespace

import cv2
import numpy as np

from dental_modes import TOOTH_TYPES, fdi_name


@dataclass
class Identity:
    votes: deque = field(default_factory=lambda: deque(maxlen=5))

    def observe(self, row):
        if row.get('identity_status') == 'conflicting':
            self.votes.clear()
        else:
            self.votes.append((row.get('tooth_type'), row.get('fdi_number'), row['confidence']))

    def label(self):
        types, numbers = Counter(), Counter()
        for kind, number, confidence in self.votes:
            if kind:
                types[kind] += confidence
            if number and confidence >= .4:
                numbers[number] += 1
        kind = types.most_common(1)[0][0] if types else None
        if numbers:
            number, hits = numbers.most_common(1)[0]
            if hits >= 2 and hits / sum(numbers.values()) >= .67 and kind == TOOTH_TYPES[int(number[1]) - 1]:
                return {'label': f'FDI {number} · {fdi_name(number)}', 'tooth_type': kind, 'fdi_number': number}
        if kind:
            return {'label': kind, 'tooth_type': kind}
        return {'label': 'Tooth · identity uncertain'}


class ExamTracker:
    def __init__(self, fps):
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.args = SimpleNamespace(track_high_thresh=.30, track_low_thresh=.1, new_track_thresh=.30,
                                    track_buffer=max(2, round(fps * .4)), match_thresh=.8, fuse_score=False)
        self.banks = {'T': BYTETracker(self.args), 'R': BYTETracker(self.args)}
        self.epoch = 1
        self.identity = {}
        self.last_seen = {}
        self.active = []

    def reset(self):
        for tracker in self.banks.values():
            tracker.reset()
        self.epoch += 1
        self.identity.clear()
        self.last_seen.clear()
        self.active.clear()

    def update(self, rows, size, frame_index, detected):
        from ultralytics.engine.results import Boxes

        width, height = size
        output = []
        for prefix, tracker in self.banks.items():
            candidates = [row for row in rows if
                          ('R' if row['category'] == 'tooth_region' else 'T') == prefix]
            # The ID matches geometry. FDI classes are smoothed separately from the tracker.
            data = np.asarray([[*row['box'], max(.31, row['confidence']), 0]
                               for row in candidates], dtype=np.float32).reshape(-1, 6)
            tracker.update(Boxes(data, (height, width)))
            for track in tracker.tracked_stracks:
                # Use the current measurement. Never draw a missing Kalman prediction.
                if track.frame_id != tracker.frame_id or not 0 <= track.idx < len(candidates):
                    continue
                row = candidates[int(track.idx)]
                key = f'{prefix}{self.epoch}-{track.track_id}'
                identity = self.identity.setdefault(key, Identity())
                self.last_seen[key] = frame_index
                if detected and prefix == 'T':
                    identity.observe(row)
                detail = identity.label() if prefix == 'T' else {'label': 'Visible tooth region'}
                measurement = {k: v for k, v in row.items() if k not in ('fdi_number', 'tooth_type', 'tooth_name')}
                output.append({**measurement, **detail, 'track_id': key, 'origin': 'detected' if detected else 'tracked'})
        for key, seen in list(self.last_seen.items()):
            if frame_index - seen > self.args.track_buffer + 2:
                self.last_seen.pop(key)
                self.identity.pop(key, None)
        self.active = output
        return output


def bridge(previous, current, rows):
    """Move boxes only when several features agree in both flow directions."""
    if not rows:
        return []
    height, width = current.shape
    groups, batches = [], []
    for row in rows:
        x1, y1, x2, y2 = map(round, row['box'])
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        region = previous[y1:y2, x1:x2]
        points = cv2.goodFeaturesToTrack(region, maxCorners=16, qualityLevel=.025, minDistance=3) if min(region.shape) >= 8 else None
        count = len(points) if points is not None and len(points) >= 4 else 0
        groups.append(count)
        if count:
            batches.append(points + np.array([x1, y1], dtype=np.float32))
    if not batches:
        return []
    points = np.concatenate(batches)
    flowed, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None,
                                                  winSize=(21, 21), maxLevel=2)
    if flowed is None:
        return []
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, flowed, None,
                                                       winSize=(21, 21), maxLevel=2)
    if backward is None:
        return []
    valid = (status.ravel() == 1) & (back_status.ravel() == 1)
    valid &= np.linalg.norm((points - backward).reshape(-1, 2), axis=1) < 1.8
    shift = (flowed - points).reshape(-1, 2)
    offset, propagated = 0, []
    for row, count in zip(rows, groups):
        keep = valid[offset:offset + count]
        delta = shift[offset:offset + count][keep]
        offset += count
        if len(delta) < max(4, count * .55):
            continue
        dx, dy = np.median(delta, axis=0)
        if np.median(np.linalg.norm(delta - [dx, dy], axis=1)) > 2.5 or np.hypot(dx, dy) > width * .08:
            continue
        x1, y1, x2, y2 = row['box']
        moved = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        if not (0 <= moved[0] < moved[2] <= width and 0 <= moved[1] < moved[3] <= height):
            continue
        propagated.append({**row, 'box': [float(value) for value in moved]})
    return propagated
