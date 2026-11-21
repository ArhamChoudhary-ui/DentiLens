"""Submission contract derived from the supplied student guide and vocabulary.

The organizer's validate.py was absent. This module is a local implementation
of the written rules, and cannot establish parity with that missing script.
"""
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'reference' / 'student_package'
if not PACKAGE.is_dir():
    PACKAGE = ROOT / 'code' / 'reference' / 'student_package'


def configured_team_id():
    value = os.environ.get('DENTILENS_TEAM_ID', '').strip()
    if value:
        return value
    path = ROOT / 'team_id.txt'
    if not path.is_file():
        path = ROOT / 'code' / 'team_id.txt'
    return path.read_text().strip() if path.is_file() else None


@dataclass(frozen=True)
class ManifestEntry:
    video_id: str
    duration_sec: float
    dataset: str


class StudentKit:
    def __init__(self, directory=PACKAGE):
        self.directory = Path(directory)
        classes = json.loads((self.directory / 'supported_classes.json').read_text())
        self.labels = set(classes['labels'])
        self.verbs = set(classes['verbs'])
        self.nouns = set(classes['nouns'])
        self.entries = {}
        with (self.directory / 'manifest.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                entry = ManifestEntry(row['video_id'], float(row['duration_sec']), row['dataset'])
                if entry.video_id in self.entries or not math.isfinite(entry.duration_sec) or entry.duration_sec <= 0:
                    raise ValueError('Invalid or duplicate manifest entry')
                self.entries[entry.video_id] = entry

    def validate_classes(self, classes):
        for key, supported in [('verb', self.verbs), ('noun', self.nouns), ('label', self.labels)]:
            unknown = set(classes[key]) - supported
            if unknown:
                raise ValueError(f'Unsupported {key} classes: {sorted(unknown)}')


def require_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')


def require_score(value, name):
    require_number(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f'{name} must be in [0, 1]')


def segment_bounds(item, duration):
    segment = item.get('segment')
    if not isinstance(segment, list) or len(segment) != 2:
        raise ValueError('segment must be a two-element array')
    start, end = segment
    require_number(start, 'segment start')
    require_number(end, 'segment end')
    if not 0 <= start < end <= duration + 0.5:
        raise ValueError('Segment outside allowed timestamp bounds')
    return start, end


def validate_submission(result, kit=None, require_manifest=True, filename=None, team_id=None):
    kit = kit or StudentKit()
    required = {'team_id', 'video_id', 'duration_sec', 'label', 'actions', 'highlights'}
    if not isinstance(result, dict) or set(result) != required:
        raise ValueError('Expected the six student-guide submission fields')
    for key in ('team_id', 'video_id'):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f'{key} must be a nonempty string')
    if team_id is not None and result['team_id'] != team_id:
        raise ValueError('Team ID differs from the configured team')
    if filename is not None and Path(filename).name != result['video_id'] + '.json':
        raise ValueError('Filename must be <video_id>.json')
    duration = result['duration_sec']
    require_number(duration, 'duration_sec')
    if duration <= 0:
        raise ValueError('duration_sec must be positive')
    entry = kit.entries.get(result['video_id'])
    if require_manifest and entry is None:
        raise ValueError('video_id is absent from the official manifest')
    if require_manifest and abs(duration - entry.duration_sec) > 0.5 + 1e-9:
        raise ValueError('duration_sec differs from the manifest by more than 0.5s')
    label = result['label']
    if not isinstance(label, dict) or set(label) != {'name', 'score'}:
        raise ValueError('label must contain name and score')
    if not isinstance(label['name'], str) or label['name'] not in kit.labels:
        raise ValueError('Unsupported video label')
    require_score(label['score'], 'label score')
    if require_manifest and entry.dataset == 'epic_kitchens_100' and label['name'] != 'cooking':
        raise ValueError('The guide requires the cooking label for EPIC videos')
    if not isinstance(result['actions'], list) or len(result['actions']) > 2000:
        raise ValueError('actions must be an array with at most 2000 items')
    for action in result['actions']:
        if not isinstance(action, dict) or set(action) != {'segment', 'verb', 'noun', 'score'}:
            raise ValueError('Invalid action fields')
        segment_bounds(action, duration)
        if not isinstance(action['verb'], str) or action['verb'] not in kit.verbs:
            raise ValueError('Unsupported action verb')
        if not isinstance(action['noun'], str) or action['noun'] not in kit.nouns:
            raise ValueError('Unsupported action noun')
        require_score(action['score'], 'action score')
    highlights = result['highlights']
    if not isinstance(highlights, list) or not highlights:
        raise ValueError('At least one highlight is required')
    previous_end = 0.0
    lengths = []
    for highlight in highlights:
        if not isinstance(highlight, dict) or set(highlight) != {'segment'}:
            raise ValueError('Each highlight must contain only segment')
        start, end = segment_bounds(highlight, duration)
        if start < previous_end:
            raise ValueError('Highlights must be chronological and non-overlapping')
        previous_end = end
        lengths.append(end - start)
    if max(lengths) < 4.0 - 1e-9:
        raise ValueError('At least one highlight must last 4.0 seconds')
    warnings = []
    if any(length < 4.0 - 1e-9 for length in lengths):
        warnings.append('Highlights shorter than 4.0 seconds are ignored during scoring')
    if sum(lengths) > 0.15 * duration + 1e-8:
        warnings.append('Highlights exceed the 15% target and incur a scoring penalty')
    return warnings


def to_submission(review, report, team_id, kit=None, require_manifest=True, filename=None):
    kit = kit or StudentKit()
    entry = kit.entries.get(review['video_id'])
    name = review['label']
    epic_category = entry is not None and entry.dataset == 'epic_kitchens_100'
    if report['mode'] == 'model' and not epic_category and 'label_score' not in report:
        raise ValueError('Category confidence is missing from this stored analysis. Re-run model inference to export it.')
    score = report.get('label_score', 0.0)
    if epic_category:
        # This is the dataset category prescribed by the guide, not an inferred action.
        name, score = 'cooking', 1.0
    elif name == 'unclassified' and report['mode'] != 'model':
        name, score = 'other', 0.0
    scores = {(a['start'], a['end'], a['verb'], a['noun']): a['confidence']
              for a in report.get('action_scores', [])}
    actions = []
    for action in review['actions']:
        key = (action['start'], action['end'], action['verb'], action['noun'])
        if key not in scores:
            raise ValueError('Action confidence is missing from the model provenance')
        actions.append({'segment': [round(action['start'], 2), round(action['end'], 2)],
                        'verb': action['verb'], 'noun': action['noun'], 'score': scores[key]})
    result = {'team_id': team_id, 'video_id': review['video_id'],
              'duration_sec': round(report['duration'], 2),
              'label': {'name': name, 'score': score}, 'actions': actions,
              'highlights': [{'segment': [round(h['start'], 2), round(h['end'], 2)]}
                             for h in review['highlights']]}
    warnings = validate_submission(result, kit, require_manifest, filename, team_id)
    return result, warnings
