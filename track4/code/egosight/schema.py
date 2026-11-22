"""Validation of the internal dashboard representation. See contract.py for submission JSON."""
import json
import math
import os
from pathlib import Path
import tempfile


def numeric(x):
    return type(x) in (int, float) and math.isfinite(x)


def validate(result, duration, vocabulary=None):
    if set(result) != {'video_id', 'label', 'actions', 'highlights'}:
        raise ValueError("Expected four internal review fields")
    if not numeric(duration) or duration <= 0:
        raise ValueError("Invalid video duration")
    for key in ('video_id', 'label'):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    for kind in ('actions', 'highlights'):
        if not isinstance(result[kind], list):
            raise ValueError(f"{kind} must be a list")
        previous_start, previous_end = -1., -1.
        for item in result[kind]:
            if not isinstance(item, dict):
                raise ValueError("Segments must be objects")
            expected = {'start','end','verb','noun'} if kind == 'actions' else {'start','end','importance'}
            if set(item) != expected:
                raise ValueError(f"Unexpected {kind} fields")
            start, end = item['start'], item['end']
            if not numeric(start) or not numeric(end) or not 0 <= start < end <= duration:
                raise ValueError("Invalid timestamp bounds")
            if start < previous_start or (kind == 'highlights' and start < previous_end):
                raise ValueError("Segments must be chronological; highlights must not overlap")
            previous_start, previous_end = start, end
            if kind == 'highlights':
                if not numeric(item['importance']) or not 0 <= item['importance'] <= 1:
                    raise ValueError("Importance must be finite in [0, 1]")
            else:
                for key in ('verb','noun'):
                    if not isinstance(item[key], str) or not item[key].strip():
                        raise ValueError("Empty action class")
                if vocabulary is not None and (item['verb'],item['noun']) not in vocabulary:
                    raise ValueError("Action outside supplied vocabulary")
    selected_seconds = sum(h['end']-h['start'] for h in result['highlights'])
    unavoidable_minimum = (duration < 4 / .15 and len(result['highlights']) == 1
                           and abs(selected_seconds - 4) < 1e-8)
    if selected_seconds > .15*duration + 1e-8 and not unavoidable_minimum:
        raise ValueError("Highlight duration exceeds 15%")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as f:
            name = f.name
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)
