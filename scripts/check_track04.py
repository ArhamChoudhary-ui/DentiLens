"""Check Track 4 output against the supplied manifest and student guide."""
import argparse
import csv
import json
import math
from pathlib import Path

KIT = Path(__file__).resolve().parents[1] / 'reference/student_package'
FIELDS = {'team_id', 'video_id', 'duration_sec', 'label', 'actions', 'highlights'}


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def require(condition, text):
    if not condition:
        raise ValueError(text)


def check(path, vocabulary, entry=None):
    data = json.loads(path.read_text())
    require(type(data) is dict and set(data) == FIELDS, 'Expected the six top-level fields.')
    team, duration = data['team_id'], data['duration_sec']
    require(isinstance(team, str) and team.strip(), 'Missing team ID.')
    require(data['video_id'] == path.stem, 'Video ID must equal the filename.')
    require(numeric(duration) and duration > 0, 'Invalid video duration.')
    if entry:
        require(abs(duration - float(entry['duration_sec'])) <= .5, 'Duration differs from the manifest.')
    label = data['label']
    require(type(label) is dict and set(label) == {'name', 'score'}, 'Invalid video label.')
    require(label['name'] in vocabulary['labels'], 'Video label is outside the vocabulary.')
    require(numeric(label['score']) and 0 <= label['score'] <= 1, 'Invalid video label score.')
    if entry and entry['dataset'] == 'epic_kitchens_100':
        require(label['name'] == 'cooking', 'EPIC-KITCHENS videos require the cooking label.')
    elif entry:
        require(label['name'] != 'cooking', 'FPVSum requires an FPVSum label or other.')

    def segment(item, fields):
        require(type(item) is dict and set(item) == fields, 'Invalid segment record.')
        pair = item['segment']
        require(type(pair) is list and len(pair) == 2 and all(numeric(i) for i in pair)
                and 0 <= pair[0] < pair[1] <= duration + .5, 'Invalid segment bounds.')
        return pair

    require(type(data['actions']) is list and len(data['actions']) <= 2000, 'Invalid actions list.')
    for action in data['actions']:
        segment(action, {'segment', 'verb', 'noun', 'score'})
        require(action['verb'] in vocabulary['verbs'] and action['noun'] in vocabulary['nouns'],
                'Action uses a class outside the vocabulary.')
        require(numeric(action['score']) and 0 <= action['score'] <= 1, 'Invalid action score.')
    require(type(data['highlights']) is list and data['highlights'], 'At least one highlight is required.')
    highlights = [segment(item, {'segment'}) for item in data['highlights']]
    require(all(a[1] <= b[0] for a, b in zip(highlights, highlights[1:])), 'Highlights overlap or are out of order.')
    lengths = [round(b - a, 6) for a, b in highlights]
    require(max(lengths) >= 4, 'At least one highlight must be 4 seconds or longer.')
    warnings = []
    if min(lengths) < 4:
        warnings.append('Highlight shorter than 4 seconds.')
    if sum(lengths) > .15 * duration + 1e-6:
        warnings.append('Highlights exceed the 15% target.')
    return team, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--examples', action='store_true')
    source.add_argument('--outputs', type=Path)
    args = parser.parse_args()
    with (KIT / 'manifest.csv').open() as stream:
        manifest = {row['video_id']: row for row in csv.DictReader(stream)}
    vocabulary = json.loads((KIT / 'supported_classes.json').read_text())
    paths = ([KIT / name for name in ('P02_109.json', 'skiing_03.json')] if args.examples
             else sorted(args.outputs.glob('*.json')))
    errors, warnings, teams = {}, {}, set()
    for path in paths:
        try:
            if not args.examples:
                require(path.stem in manifest, 'Video is outside the manifest.')
            team, flags = check(path, vocabulary, None if args.examples else manifest[path.stem])
            teams.add(team)
            if flags:
                warnings[path.stem] = flags
        except (ValueError, OSError) as exc:
            errors[path.stem] = str(exc)
    if len(teams) > 1:
        errors['team_id'] = 'Team ID must match across files.'
    missing = [] if args.examples else sorted(manifest.keys() - {path.stem for path in paths})
    print(json.dumps(dict(manifest_count=len(manifest), files_checked=len(paths),
                          errors=errors, warnings=warnings, missing=missing), indent=2))
    return int(bool(errors or missing))


if __name__ == '__main__':
    raise SystemExit(main())
