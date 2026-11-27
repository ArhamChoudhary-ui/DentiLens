import pytest
from egosight.selection import select, explain_highlights


def test_select_preserves_budget_and_order():
    candidates = [
        {'start': 0.0, 'end': 4.0, 'confidence': 0.9, 'visual': 0.01, 'verb': 'other', 'noun': 'other'}, # intro
        {'start': 10.0, 'end': 14.0, 'confidence': 0.85, 'visual': 0.08, 'verb': 'other', 'noun': 'other'},
        {'start': 20.0, 'end': 26.0, 'confidence': 0.80, 'visual': 0.07, 'verb': 'other', 'noun': 'other'}, # 6s candidate
        {'start': 40.0, 'end': 44.0, 'confidence': 0.75, 'visual': 0.09, 'verb': 'other', 'noun': 'other'},
    ]
    # Duration = 60s, budget = 9.0s
    evidence = []
    selected = select(candidates, duration=60.0, strategy='diversity', ensure_highlight=True, evidence=evidence)

    assert selected
    # Chronological
    assert all(a['end'] <= b['start'] for a, b in zip(selected, selected[1:]))
    # Budget <= 9.0s
    assert sum(h['end'] - h['start'] for h in selected) <= 60.0 * 0.15 + 1e-7
    # At least one highlight >= 4.0s
    assert any(h['end'] - h['start'] >= 4.0 - 1e-9 for h in selected)


def test_explain_highlights_links_proposals():
    highlights = [{'start': 10.0, 'end': 14.0, 'importance': 0.82}]
    actions = [
        {'start': 0.0, 'end': 4.0, 'confidence': 0.5, 'verb': 'other', 'noun': 'other'},
        {'start': 9.0, 'end': 13.0, 'confidence': 0.85, 'verb': 'other', 'noun': 'other'}, # Proposal #2 overlaps
    ]
    details = explain_highlights(highlights, actions, activity='rockclimbing')
    assert len(details) == 1
    assert details[0]['related_proposal_index'] == 2
    assert 'rockclimbing' in details[0]['rationale']
    assert details[0]['duration'] == 4.0
