"""Tests for AI Scene Understanding and Visual Narration."""
from pathlib import Path
import pytest

from egosight.scene_understanding import (
    BlipCaptioner,
    format_clock,
    sample_scene_timestamps,
    analyze_scene_understanding,
)


def test_format_clock():
    assert format_clock(0.0) == "00:00"
    assert format_clock(5.4) == "00:05"
    assert format_clock(65.0) == "01:05"
    assert format_clock(125.8) == "02:05"


def test_clean_caption():
    clean = BlipCaptioner._clean_caption
    assert clean("a person climbing a rock") == "A person climbing a rock."
    assert clean("A person climbing a rock.") == "A person climbing a rock."
    assert clean("") == "Visual scene content detected."


def test_sample_scene_timestamps():
    # Test normal sampling
    duration = 60.0
    highlights = [{"segment": [10.0, 14.0]}, {"segment": [40.0, 48.0]}]
    actions = [
        {"segment": [5.0, 8.0], "score": 0.9},
        {"segment": [20.0, 24.0], "score": 0.8},
        {"segment": [30.0, 34.0], "score": 0.7},
    ]
    sampled = sample_scene_timestamps(duration, highlights, actions, target_count=8, min_gap=2.0)
    assert len(sampled) <= 8
    assert len(sampled) > 0

    # Ensure strictly sorted chronologically
    timestamps = [s["timestamp"] for s in sampled]
    assert timestamps == sorted(timestamps)

    # Ensure all timestamps are within bounds
    for ts in timestamps:
        assert 0.2 <= ts <= duration - 0.2

    # Ensure min gap spacing
    for i in range(len(timestamps) - 1):
        assert timestamps[i + 1] - timestamps[i] >= 1.99


def test_sample_scene_timestamps_empty():
    assert sample_scene_timestamps(0.0) == []
    assert sample_scene_timestamps(-5.0) == []


def test_analyze_scene_understanding_missing_video(tmp_path):
    res = analyze_scene_understanding(
        video_path=tmp_path / "non_existent.mp4",
        output_dir=tmp_path,
    )
    assert res["status"] == "failed"
    assert "not found" in res["error"].lower()
    assert res["scenes"] == []
