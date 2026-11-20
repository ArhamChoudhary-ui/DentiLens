"""Regression tests for exact observation ranges and organizer-kit checks."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'scripts'))
from exam_intelligence import appearances, search_exam
from check_track04 import KIT, check


def test_an_appearance_never_bridges_a_missing_frame(tmp_path):
    rows = [{'frame': frame, 'time': frame / 10} for frame in (2, 3, 7)]
    assert [span['count'] for span in appearances(rows)] == [2, 1]
    timeline = tmp_path / 'timeline.jsonl'
    with timeline.open('w') as stream:
        for frame in rows:
            annotations = [{'category': 'tooth_identity', 'track_id': 'T1-1',
                            'fdi_number': '21', 'label': 'FDI 21 · upper left central incisor'}]
            stream.write(json.dumps(dict(frame, annotations=annotations)) + '\n')
    report = {'findings': [], 'video': {'fps': 10}}
    assert search_exam(timeline, report, [], 'tooth 21 at 00:00.40')['results'] == []
    result = search_exam(timeline, report, [], 'tooth 21 at 00:00.70')['results']
    assert result[0]['time'] == .7 and result[0]['status'] == 'suggested'


def test_supplied_track04_examples_expose_label_discrepancy():
    vocabulary = json.loads((KIT / 'supported_classes.json').read_text())
    assert check(KIT / 'P02_109.json', vocabulary) == ('T07', [])
    with pytest.raises(ValueError, match='outside the vocabulary'):
        check(KIT / 'skiing_03.json', vocabulary)
