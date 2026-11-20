"""Verify saved timing, mask geometry and highlight assets, not clinical accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def verify(video, analysis):
    video, analysis = Path(video), Path(analysis)
    report = json.loads((analysis / 'report.json').read_text())
    with video.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if report.get('input_sha256') and report['input_sha256'] != digest:
        raise ValueError('Source video hash does not match the report')
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError('Source video could not be opened')
        fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        assert frames == report['frames_decoded']
        assert abs(fps - report['fps']) < 1e-6
        assert abs(frames/fps - report['duration_sec']) < 1e-6
    finally:
        cap.release()
    for item in report['observations']:
        assert 0 <= item['source_frame'] < frames
        assert abs(item['timestamp_sec'] - item['source_frame']/fps) < 1e-6
        assert 0 <= item['confidence'] <= 1
        x1, y1, x2, y2 = item['bbox_xyxy_normalized']
        assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
        assert cv2.imread(str(analysis / item['evidence'])) is not None
        if 'mask_path' in item:
            mask = cv2.imread(str(analysis / item['mask_path']), cv2.IMREAD_GRAYSCALE)
            assert mask is not None and [mask.shape[1], mask.shape[0]] == item['mask_size']
            assert set(np.unique(mask)).issubset({0, 255}) and mask.any()
            assert item['source_size'] == size
    total, end = 0.0, 0.0
    for clip in report['highlights']:
        assert 0 <= clip['start_sec'] < clip['end_sec'] <= report['duration_sec']
        assert clip['start_sec'] >= end
        assert clip['end_sec'] - clip['start_sec'] >= 4 - 1e-6
        total += clip['end_sec'] - clip['start_sec']
        end = clip['end_sec']
    assert total <= report['duration_sec'] * 0.15 + 1e-6
    assert abs(total - report['highlight_duration_sec']) < 1e-6
    if total:
        cap = cv2.VideoCapture(str(analysis / 'highlights.mp4'))
        try:
            assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) > 0
            actual = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            assert abs(actual-total) <= 1/cap.get(cv2.CAP_PROP_FPS) + 1e-6
        finally:
            cap.release()
    return {'input_sha256': digest, 'frames_sampled': report['sampled_frames'],
            'frames_with_predictions': len({o['source_frame'] for o in report['observations']}),
            'observations': len(report['observations']), 'masks_checked': sum('mask_path' in o for o in report['observations']),
            'highlight_duration_sec': total, 'verification': 'Source hash, timing, mask assets and highlight geometry checked.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('analysis', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.video, args.analysis), indent=2))
