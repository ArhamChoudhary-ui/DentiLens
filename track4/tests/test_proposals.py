import pytest
from egosight.proposals import tiou, categorize_proposal, process_proposals, proposal_rationale


def test_tiou_calculations():
    a = {'start': 0.0, 'end': 4.0}
    b = {'start': 2.0, 'end': 6.0}
    # Intersection = 2.0, Union = 6.0 -> 1/3
    assert abs(tiou(a, b) - 1.0 / 3.0) < 1e-6
    # Non-overlapping
    c = {'start': 4.0, 'end': 8.0}
    assert tiou(a, c) == 0.0


def test_categorize_proposal_logic():
    # 1. Genuine recognized action
    assert categorize_proposal('cut', 'onion', 0.9, 0.05, 0.0, 0.0) == 'Recognized action'
    # 2. Activity transition (gradient >= 0.15)
    assert categorize_proposal('other', 'other', 0.4, 0.05, 0.2, 0.01) == 'Activity transition'
    # 3. Potential activity interval (high conf and motion)
    assert categorize_proposal('other', 'other', 0.45, 0.05, 0.02, 0.01) == 'Potential activity interval'
    # 4. Visually unclear interval (low motion or low confidence)
    assert categorize_proposal('other', 'other', 0.15, 0.01, 0.0, 0.0) == 'Visually unclear interval'


def test_process_proposals_nms_suppression():
    # Create 5 contiguous overlapping windows
    # Window 0: 0-4s, score 0.35, motion 0.04
    # Window 1: 2-6s, score 0.85, motion 0.08  <-- Peak
    # Window 2: 4-8s, score 0.40, motion 0.04
    # Window 3: 10-14s, score 0.70, motion 0.07 <-- Peak 2
    # Window 4: 12-16s, score 0.20, motion 0.02
    window_scores = [
        (0.0, 4.0, {'rockclimbing': 0.35}, 0.04),
        (2.0, 6.0, {'rockclimbing': 0.85}, 0.08),
        (4.0, 8.0, {'rockclimbing': 0.40}, 0.04),
        (10.0, 14.0, {'rockclimbing': 0.70}, 0.07),
        (12.0, 16.0, {'rockclimbing': 0.20}, 0.02),
    ]
    actions, metadata, stats = process_proposals(window_scores, duration=20.0, label='rockclimbing', iou_thresh=0.35)

    assert stats['raw_candidates'] == 5
    # The peak window 2.0-6.0 overlaps with 0-4 and 4-8 (tiou = 2/6 = 0.333 or 2/6, if iou_thresh=0.30 it suppresses both)
    # Check that peak proposals are retained and sorted chronologically
    assert len(actions) >= 2
    assert actions[0]['start'] < actions[1]['start']
    assert all('confidence' in a and 'visual' in a for a in actions)
    assert all('category' in m and 'quality_rank' in m and 'rationale' in m for m in metadata)


def test_process_proposals_custom_actions_epic():
    custom = [
        {'start': 1.0, 'end': 3.0, 'verb': 'take', 'noun': 'knife', 'confidence': 0.88, 'visual': 0.05},
        {'start': 4.0, 'end': 7.0, 'verb': 'cut', 'noun': 'bread', 'confidence': 0.92, 'visual': 0.08},
    ]
    actions, metadata, stats = process_proposals([], duration=10.0, label='cooking', custom_actions=custom)
    assert len(actions) == 2
    assert metadata[0]['category'] == 'Recognized action'
    assert metadata[1]['category'] == 'Recognized action'
    assert 'knife' in metadata[0]['rationale'] or 'Recognized' in metadata[0]['rationale']
