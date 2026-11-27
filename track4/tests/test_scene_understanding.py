"""Tests for upgraded AI Scene Understanding and Temporal Visual Narration."""
import numpy as np
import pytest

from egosight.scene_understanding import (
    BlipCaptioner,
    apply_temporal_consistency,
    format_clock,
    inspect_frame_quality,
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
    assert clean("A photo of a person climbing a rock.") == "A person climbing a rock."
    assert clean("the video shows a wave") == "A wave."
    assert clean("") == "Visual scene content detected."


def test_inspect_frame_quality():
    # 1. Title screen (mostly black with white text)
    title_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    title_frame[20:80, 20:80] = 255  # 36% white, 64% black
    cat, override, quality, _ = inspect_frame_quality(title_frame)
    assert cat == "title_screen"
    assert "title card" in override.lower()
    assert quality == "Title / intro screen"

    # 2. Dark frame
    dark_frame = np.full((100, 100, 3), 10, dtype=np.uint8)
    cat, override, quality, _ = inspect_frame_quality(dark_frame)
    assert cat == "dark_frame"
    assert "dark transition" in override.lower()

    # 3. Submerged water frame (green/cyan, high saturation)
    water_hsv = np.zeros((100, 100, 3), dtype=np.uint8)
    water_hsv[:, :, 0] = 80  # green/cyan hue
    water_hsv[:, :, 1] = 180  # high saturation
    water_hsv[:, :, 2] = 140
    import cv2
    water_bgr = cv2.cvtColor(water_hsv, cv2.COLOR_HSV2BGR)
    cat, override, quality, _ = inspect_frame_quality(water_bgr)
    assert cat == "submerged_water"
    assert "submerged camera view" in override.lower()


def test_apply_temporal_consistency():
    scenes = [
        {"caption": "A person riding a wave on a surfboard.", "quality": "Clear visual description"},
        {"caption": "A fish is swimming in the water.", "quality": "Clear visual description"},
        {"caption": "A large green cloud in the sky.", "quality": "Clear visual description"},
    ]
    metrics = [
        {"water_ratio": 0.8, "lap_var": 20.0, "mean_gray": 120.0},
        {"water_ratio": 0.75, "lap_var": 15.0, "mean_gray": 160.0},
        {"water_ratio": 0.95, "lap_var": 3.0, "mean_gray": 140.0},
    ]
    consistent = apply_temporal_consistency(scenes, metrics)
    assert "ocean wave breaks" in consistent[1]["caption"].lower()
    assert consistent[1]["quality"] == "Uncertain interpretation"
    assert "turbulent green ocean water" in consistent[2]["caption"].lower()
    assert consistent[2]["quality"] == "Limited visual information"

    # Aerial context test
    aerial_scenes = [
        {"caption": "A skateboarder ' s view of a skate park.", "quality": "Clear visual description"},
        {"caption": "A view of a small town from a helicopter.", "quality": "Clear visual description"},
        {"caption": "A person is flying a kite in the air.", "quality": "Clear visual description"},
    ]
    aerial_metrics = [{}, {}, {}]
    aerial_consistent = apply_temporal_consistency(aerial_scenes, aerial_metrics)
    assert "takeoff pad" in aerial_consistent[0]["caption"].lower()
    assert "rotor blur" in aerial_consistent[2]["caption"].lower()


def test_sample_scene_timestamps():
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

    timestamps = [s["timestamp"] for s in sampled]
    assert timestamps == sorted(timestamps)
    for ts in timestamps:
        assert 0.2 <= ts <= duration - 0.2


def test_analyze_scene_understanding_missing_video(tmp_path):
    res = analyze_scene_understanding(
        video_path=tmp_path / "non_existent.mp4",
        output_dir=tmp_path,
    )
    assert res["status"] == "failed"
    assert "not found" in res["error"].lower()
    assert res["scenes"] == []
