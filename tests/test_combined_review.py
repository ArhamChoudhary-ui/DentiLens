"""Test review rules for overlapping model predictions."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from combined_review import combine_identities, nearby_identity


def detection(label, bbox, score=.8):
    return {'model_label': label, 'bbox_xyxy_normalized': bbox, 'confidence': score}


def test_one_tooth_shows_name_number_and_type_without_duplicate_labels():
    box = [.1, .2, .3, .4]
    numbers = [detection('11', box), detection('11', box, .5)]
    types = [detection('central incisor', box), detection('canine', [.5, .2, .6, .4])]
    matched, unmatched = combine_identities(numbers, types)
    assert matched['fdi_number'] == '11'
    assert matched['tooth_name'] == 'upper right central incisor'
    assert matched['tooth_type'] == 'central incisor'
    assert matched['identity_status'] == 'type agreement'
    assert unmatched['tooth_type'] == 'canine'
    assert 'fdi_number' not in unmatched


def test_conflicting_type_withholds_exact_number_and_cavity_association():
    box = [.1, .2, .3, .4]
    conflict, = combine_identities([detection('11', box)], [detection('canine', box)])
    assert 'fdi_number' not in conflict
    assert conflict['fdi_candidate'] == '11'
    assert conflict['type_candidate'] == 'canine'
    assert conflict['identity_status'] == 'conflicting'
    assert nearby_identity(detection('possible cavity', box), [conflict]) is None


def test_cavity_location_suggestion_requires_one_clear_overlap():
    teeth = combine_identities([detection('11', [.1, .1, .3, .4]),
                                detection('21', [.3, .1, .5, .4])], [])
    assert nearby_identity(detection('possible cavity', [.11, .2, .27, .3]), teeth).startswith('FDI 11')
    assert nearby_identity(detection('possible cavity', [.22, .2, .4, .3]), teeth) is None
