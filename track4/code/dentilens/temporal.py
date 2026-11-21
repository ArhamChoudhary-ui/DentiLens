"""Temporal postprocessing of model-derived window scores."""
import numpy as np


def tiou(a, b):
    intersection = max(0.0, min(a['end'], b['end']) - max(a['start'], b['start']))
    union = max(a['end'], b['end']) - min(a['start'], b['start'])
    return intersection / union if union > 0 else 0.0


def smooth(scores, radius=1):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all() or radius < 0:
        raise ValueError("Expected finite window-by-class scores")
    return np.array([scores[max(0, i-radius):i+radius+1].mean(axis=0)
                     for i in range(len(scores))])


def nms(events, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid NMS threshold")
    retained = []
    for event in sorted(events, key=lambda x: (-x['confidence'], x['start'])):
        if all((event['verb'], event['noun']) != (r['verb'], r['noun'])
               or tiou(event, r) <= threshold for r in retained):
            retained.append(event)
    return sorted(retained, key=lambda x: (x['start'], x['end']))
