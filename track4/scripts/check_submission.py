"""Local checks derived from the student guide. Organizer validate.py is absent."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from dentilens.contract import StudentKit, validate_submission
from dentilens.schema import atomic_json


EXCLUDED_BY_ORGANIZER = frozenset({'Scuba_KD74Ywuy9ng', 'Scuba_Um4kqB8bQBg'})


def check_files(paths, kit, require_manifest=True, team_id=None):
    errors, warnings, seen = {}, {}, set()
    for path in paths:
        try:
            result = json.loads(path.read_text())
            file_warnings = validate_submission(result, kit, require_manifest, path, team_id)
            if team_id is None:
                team_id = result['team_id']
            if result['video_id'] in seen:
                raise ValueError('Duplicate video ID')
            seen.add(result['video_id'])
            if file_warnings:
                warnings[path.name] = file_warnings
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors[path.name] = str(exc)
    missing = set(kit.entries) - seen
    excluded = EXCLUDED_BY_ORGANIZER & set(kit.entries)
    return {'required_missing_video_ids': sorted(missing - excluded),
            'organizer_excluded_video_ids': sorted(excluded),
            'team_id': team_id, 'valid_videos': len(seen), 'errors': errors,
            'warnings': warnings, 'missing_video_ids': sorted(missing),
            'organizer_validator_available': (kit.directory / 'validate.py').is_file(),
            'validator_source': 'Local implementation of supplied student guide'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--file', type=Path)
    inputs.add_argument('--dir', type=Path)
    parser.add_argument('--team-id')
    parser.add_argument('--schema-only', action='store_true', help='Ignore manifest membership/duration for examples')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.dir and not args.dir.is_dir():
        parser.error('Output directory does not exist')
    paths = [args.file] if args.file else sorted(args.dir.glob('*.json'))
    report = check_files(paths, StudentKit(), not args.schema_only, args.team_id)
    if args.file is not None or args.schema_only:
        report['missing_video_ids'] = []
        report['required_missing_video_ids'] = []
    if args.report:
        atomic_json(args.report, report)
    print(json.dumps(report, indent=2))
    failed = bool(report['errors']) or (args.dir is not None and not args.schema_only and bool(report['required_missing_video_ids']))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
