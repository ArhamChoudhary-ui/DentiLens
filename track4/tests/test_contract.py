"""Contract regression fixtures use synthetic scores/media, never benchmark evidence."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from egosight.contract import StudentKit, to_submission, validate_submission
from egosight.pipeline import analyze
from egosight.selection import select
from make_test_video import make_video
from check_submission import check_files


@pytest.fixture
def kit():
    return StudentKit()


@pytest.fixture
def submission(kit):
    # Use an actual manifest ID/duration solely to test contract validation.
    return {'team_id': 'test_team', 'video_id': 'P28_18', 'duration_sec': 120.27,
            'label': {'name': 'cooking', 'score': 1.0},
            'actions': [{'segment': [0, 4], 'verb': 'wash', 'noun': 'cup', 'score': 0.6}],
            'highlights': [{'segment': [0, 4]}]}


def test_official_kit_counts_and_duration_range(kit):
    assert len(kit.entries) == 187
    assert (len(kit.labels), len(kit.verbs), len(kit.nouns)) == (16, 98, 301)
    assert min(entry.duration_sec for entry in kit.entries.values()) == 11.13
    assert max(entry.duration_sec for entry in kit.entries.values()) == 1968.6


def test_submission_schema_and_filename(submission, kit):
    assert validate_submission(submission, kit, filename='P28_18.json') == []
    with pytest.raises(ValueError, match='Filename'):
        validate_submission(submission, kit, filename='other.json')


@pytest.mark.parametrize('value', [True, '0.7', float('nan'), float('inf'), -0.1, 1.1])
def test_score_types_and_bounds(submission, kit, value):
    submission['actions'][0]['score'] = value
    with pytest.raises(ValueError):
        validate_submission(submission, kit)


def test_manifest_duration_tolerance(submission, kit):
    submission['duration_sec'] += 0.5
    validate_submission(submission, kit)
    submission['duration_sec'] += 0.01
    with pytest.raises(ValueError, match='duration_sec differs'):
        validate_submission(submission, kit)


@pytest.mark.parametrize('field,value', [('verb', 'examining'), ('noun', 'tooth')])
def test_reject_untrained_taxonomy_mapping(submission, kit, field, value):
    submission['actions'][0][field] = value
    with pytest.raises(ValueError, match='Unsupported action'):
        validate_submission(submission, kit)


def test_action_limit(submission, kit):
    submission['actions'] *= 2001
    with pytest.raises(ValueError, match='2000'):
        validate_submission(submission, kit)


def test_highlight_minimum_and_budget_warning(submission, kit):
    submission['highlights'] = [{'segment': [0, 3.99]}]
    with pytest.raises(ValueError, match='4.0 seconds'):
        validate_submission(submission, kit)
    submission['highlights'] = [{'segment': [0, 20]}]
    assert any('15%' in warning for warning in validate_submission(submission, kit))


def test_overlapping_highlights_rejected(submission, kit):
    submission['highlights'] = [{'segment': [0, 4]}, {'segment': [3, 7]}]
    with pytest.raises(ValueError, match='non-overlapping'):
        validate_submission(submission, kit)


def test_short_tail_candidates_are_excluded():
    candidates = [{'start': 0, 'end': 2, 'confidence': 1},
                  {'start': 10, 'end': 14, 'confidence': 0.8}]
    assert select(candidates, 120, ensure_highlight=True)[0]['start'] == 10


@pytest.mark.parametrize('video_id', ['P03_26', 'P26_30', 'P28_23'])
def test_unavoidable_short_manifest_budget_exception(video_id, kit):
    duration = kit.entries[video_id].duration_sec
    highlights = select([{'start': 0, 'end': 4, 'confidence': 0.5}], duration, ensure_highlight=True)
    assert sum(h['end'] - h['start'] for h in highlights) == 4
    result = {'team_id': 'test_team', 'video_id': video_id, 'duration_sec': duration,
              'label': {'name': 'cooking', 'score': 1}, 'actions': [],
              'highlights': [{'segment': [0, 4]}]}
    assert any('15%' in warning for warning in validate_submission(result, kit))


def test_supplied_examples_are_not_manifest_outputs(kit):
    for filename in ['P02_109.json', 'skiing_03.json']:
        result = json.loads((kit.directory / filename).read_text())
        with pytest.raises(ValueError, match='absent'):
            validate_submission(result, kit)
    result = json.loads((kit.directory / 'P02_109.json').read_text())
    assert validate_submission(result, kit, require_manifest=False) == []
    result = json.loads((kit.directory / 'skiing_03.json').read_text())
    assert result['label']['name'] == 'ski'
    assert validate_submission(result, kit, require_manifest=False) == []
    result['label']['name'] = 'skiing'
    with pytest.raises(ValueError, match='Unsupported video label'):
        validate_submission(result, kit, require_manifest=False)


def test_directory_checks_missing_outputs_and_mixed_teams(tmp_path, submission, kit):
    first = tmp_path / 'P28_18.json'
    first.write_text(json.dumps(submission))
    second = copy.deepcopy(submission)
    second['team_id'] = 'different_team'
    second['video_id'] = 'P01_11'
    second['duration_sec'] = kit.entries['P01_11'].duration_sec
    (tmp_path / 'P01_11.json').write_text(json.dumps(second))
    report = check_files([first, tmp_path / 'P01_11.json'], kit)
    assert report['valid_videos'] == 1
    assert 'P01_11.json' in report['errors']
    assert len(report['missing_video_ids']) == 186


def test_default_cli_emits_six_key_contract(video, tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / 'P28_18.mp4'
    shutil.copyfile(video, source)
    output = tmp_path / 'P28_18.json'
    command = ['bash', str(root / 'run.sh'), str(source), str(output),
               '--mode', 'motion-baseline', '--team-id', 'test_team']
    result = subprocess.run(command, capture_output=True, text=True,
                            env={**os.environ, 'EGOSIGHT_PYTHON': sys.executable})
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert set(payload) == {'team_id', 'video_id', 'duration_sec', 'label', 'actions', 'highlights'}
    assert payload['actions'] == []
    assert validate_submission(payload, filename=output) == []


def test_short_video_normal_pipeline(tmp_path):
    source = tmp_path / 'short.mp4'
    make_video(source, seconds=11.1)
    review, report = analyze(source, mode='motion-baseline')
    payload, warnings = to_submission(review, report, 'development_team', require_manifest=False)
    assert payload['label'] == {'name': 'other', 'score': 0.0}
    assert payload['highlights'][0]['segment'][1] - payload['highlights'][0]['segment'][0] == 4
    assert any('15%' in warning for warning in warnings)


def test_previously_rejected_long_video(tmp_path):
    source = tmp_path / 'long.mp4'
    # Low-resolution generated fixture exercises the real decode path above 10 minutes.
    make_video(source, seconds=601, fps=1)
    review, report = analyze(source, mode='infrastructure')
    assert report['duration'] == 601
    assert review['highlights'][0]['end'] - review['highlights'][0]['start'] >= 4


def test_cli_rejects_wrong_source_duration_before_inference(video, tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / 'P03_26.mp4'
    shutil.copyfile(video, source)
    output = tmp_path / 'P03_26.json'
    result = subprocess.run(['bash', str(root / 'run.sh'), str(source), str(output),
                             '--mode', 'motion-baseline', '--team-id', 'test_team'],
                            env={**os.environ, 'EGOSIGHT_PYTHON': sys.executable},
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert 'duration differs' in result.stderr
    assert not output.exists()


def test_stored_model_output_requires_real_category_confidence():
    review = {'video_id': 'Bike_lhTFVa0h9Rk', 'label': 'bike', 'actions': [],
              'highlights': [{'start': 0, 'end': 4, 'importance': 0.4}]}
    report = {'mode': 'model', 'duration': 287.51}
    with pytest.raises(ValueError, match='Category confidence is missing'):
        to_submission(review, report, 'test_team')
