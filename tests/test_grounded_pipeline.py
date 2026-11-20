"""Mechanical integration fixtures; model accuracy is evaluated separately."""
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from dental_video import analyze


class FixtureSegmenter:
    names = {0: 'teeth'}
    prompt = 'mechanical fixture'
    provenance = {'test': 'Fixed synthetic masks for serialization checks only'}

    def predict(self, frame):
        mask = np.zeros((32, 48), dtype=np.uint8)
        mask[8:24, 12:36] = 1
        return [{'model_class_id': 0, 'model_label': 'teeth', 'confidence': 0.75,
                 'bbox_xyxy_normalized': [0.25, 0.25, 0.75, 0.75],
                 'mask': mask, 'mask_quality_estimate': 0.8}], cv2.resize(frame, (48, 32))


def test_predictor_masks_preserve_timestamps_dimensions_and_source(tmp_path):
    source = tmp_path / 'synthetic.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (96, 64))
    assert writer.isOpened()
    for _ in range(40):
        writer.write(np.zeros((64, 96, 3), dtype=np.uint8))
    writer.release()
    original = source.read_bytes()
    destination = tmp_path / 'analysis'
    report = analyze(source, tmp_path / 'unused.pt', destination, predictor=FixtureSegmenter())
    assert source.read_bytes() == original
    assert report['sampled_frames'] == 4
    assert [item['source_frame'] for item in report['observations']] == [0, 10, 20, 30]
    assert [item['timestamp_sec'] for item in report['observations']] == [0, 1, 2, 3]
    for item in report['observations']:
        assert item['source_size'] == [96, 64]
        assert item['mask_size'] == [48, 32]
        mask = cv2.imread(str(destination / item['mask_path']), cv2.IMREAD_GRAYSCALE)
        assert mask.shape == (32, 48)
        assert set(np.unique(mask)) == {0, 255}
    assert (destination / 'report.json').is_file()
    assert report['capabilities']['individual_tooth_numbering'] is False
