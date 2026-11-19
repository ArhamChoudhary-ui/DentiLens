"""Synthetic fixtures verify mechanics. They provide no dental accuracy evidence."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from dental_video import pick_highlights


def observation(timestamp, label='restoration', confidence=0.8):
    return {'timestamp_sec': timestamp, 'model_label': label, 'confidence': confidence}


def test_no_detections_produces_no_invented_highlights():
    assert pick_highlights([], 120) == []


def test_observation_clips_stay_in_source_order_and_budget():
    observations = [observation(second) for second in range(120)]
    clips = pick_highlights(observations, 120)
    assert sum(clip['end_sec'] - clip['start_sec'] for clip in clips) <= 18
    assert all(a['end_sec'] <= b['start_sec'] for a, b in zip(clips, clips[1:]))


def test_last_observation_near_video_end_has_valid_source_bounds():
    clips = pick_highlights([observation(119.9)], 120)
    assert len(clips) == 1
    assert clips[0]['start_sec'] == 116
    assert clips[0]['end_sec'] == 120
    assert clips[0]['importance'] == pytest.approx(0.86)
    assert clips[0]['model_labels'] == ['restoration']


def test_short_video_does_not_exceed_budget():
    assert pick_highlights([observation(1)], 10) == []


def test_distinct_model_labels_receive_coverage_credit():
    clips = pick_highlights([observation(0, 'class-a', 0.95),
                            observation(4, 'class-a', 0.94),
                            observation(8, 'class-b', 0.85)], 60)
    assert [clip['start_sec'] for clip in clips] == [0, 8]
