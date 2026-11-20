"""Verify that playback holds the exact annotated source frame."""
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from annotated_reel import context_windows, export_annotated_reel, review_marks


def test_reel_holds_only_the_selected_source_frame(tmp_path):
    source = tmp_path / 'source.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (96, 64))
    assert writer.isOpened()
    for index in range(30):
        writer.write(np.full((64, 96, 3), 30 + index * 4, dtype=np.uint8))
    writer.release()
    observations = [
        dict(source_frame=10, timestamp_sec=1.0, fdi_number='11', tooth_name='upper right central incisor',
             bbox_xyxy_normalized=[.3, .3, .5, .7]),
        dict(source_frame=20, timestamp_sec=2.0, model_label='dental mirror head',
             bbox_xyxy_normalized=[.3, .3, .5, .7]),
    ]
    report = dict(observations=observations, fps=10, frames_decoded=30)
    marks = review_marks(report, {'model-0': {'decision': 'Confirm'}, 'model-1': {'decision': 'Dismiss'}})
    assert [mark['source_frame'] for mark in marks] == [10]
    output = tmp_path / 'reel.mp4'
    result = export_annotated_reel(source, output, report, context_windows([1], 3), marks, tmp_path)
    assert result == {'annotated_frames': 1, 'duration_sec': 3.9}

    capture = cv2.VideoCapture(str(output))
    try:
        levels = []
        overlays = []
        while True:
            ok, image = capture.read()
            if not ok:
                break
            levels.append(float(image[0:10, 0:10].mean()))
            region = image[:160, 68:126].astype(int)
            overlays.append(bool(((region[:, :, 1] - region[:, :, 0]) > 35).any()))
    finally:
        capture.release()
    assert len(levels) == 39
    assert max(levels[10:20]) - min(levels[10:20]) < 2
    assert levels[9] < levels[10] < levels[20]
    assert overlays == [10 <= index < 20 for index in range(39)]
