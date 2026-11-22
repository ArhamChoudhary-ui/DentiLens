"""Resume real model inference and audit the organizer manifest."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

from check_submission import StudentKit, atomic_json, validate_submission


ROOT = Path(__file__).resolve().parents[1]
RESULT_ERRORS = (ValueError, TypeError, OSError, KeyError)


def inspect_outputs(outputs, kit, team_id):
    valid, invalid, claims = {}, {}, defaultdict(list)
    for path in sorted(outputs.rglob('*.json')):
        relative = path.relative_to(outputs).as_posix()
        try:
            result = json.loads(path.read_text())
            if isinstance(result, dict) and isinstance(result.get('video_id'), str):
                claims[result['video_id']].append(relative)
            if path.parent != outputs:
                raise ValueError('Output must be directly inside the output directory')
            warnings = validate_submission(result, kit, filename=path, team_id=team_id)
            selected_seconds = sum(end - start for start, end in
                                   (item['segment'] for item in result['highlights']))
            valid[path.stem] = {
                'duration_sec': result['duration_sec'],
                'label': result['label'],
                'actions': len(result['actions']),
                'highlights': len(result['highlights']),
                'highlight_duration_sec': round(selected_seconds, 2),
                'highlight_percent': round(100 * selected_seconds / result['duration_sec'], 2),
                'warnings': warnings,
            }
        except RESULT_ERRORS as exc:
            invalid[relative] = str(exc)
    duplicates = {key: paths for key, paths in claims.items() if len(paths) > 1}
    return valid, invalid, duplicates


def run_manifest(videos, outputs, report, team_id, check_only=False, kit=None):
    kit = kit or StudentKit()
    if report.resolve().is_relative_to(outputs.resolve()):
        raise ValueError('Keep the audit outside the output directory')
    media, wrong_extensions = defaultdict(list), defaultdict(list)
    for path in sorted(videos.rglob('*')):
        if path.is_file() and path.stem in kit.entries:
            destination = media if path.suffix.lower() == '.mp4' else wrong_extensions
            destination[path.stem].append(path.relative_to(videos).as_posix())
    prior = {}
    if report.is_file():
        try:
            history = json.loads(report.read_text())
            if history.get('team_id') == team_id:
                prior = history.get('videos', {})
        except (ValueError, OSError, AttributeError):
            pass

    valid, invalid, duplicates = inspect_outputs(outputs, kit, team_id)
    ledger = {}
    for video_id, entry in kit.entries.items():
        existing = prior.get(video_id, {})
        inputs = media[video_id]
        row = {
            'dataset': entry.dataset,
            'expected_input_filename': video_id + '.mp4',
            'input_paths': inputs,
            'other_extensions': wrong_extensions[video_id],
            'expected_output_filename': video_id + '.json',
            'manifest_duration_sec': entry.duration_sec,
            'inference_seconds': existing.get('inference_seconds') if video_id in valid else None,
        }
        if video_id in valid:
            row.update(valid[video_id], status='reused')
        elif row['expected_output_filename'] in invalid:
            row.update(status='invalid_output', error=invalid[row['expected_output_filename']])
        elif len(inputs) > 1:
            row.update(status='duplicate_input', error='Multiple MP4 files match this ID')
        elif not inputs:
            row['status'] = 'missing_input'
        else:
            row['status'] = 'pending'
        ledger[video_id] = row

    def checkpoint():
        status_counts = Counter(row['status'] for row in ledger.values())
        audit = {
            'team_id': team_id,
            'validator_source': 'Local implementation of supplied student guide',
            'organizer_validator_available': (kit.directory / 'validate.py').is_file(),
            'expected': len(ledger),
            'expected_by_dataset': dict(Counter(row['dataset'] for row in ledger.values())),
            'available': sum(len(row['input_paths']) == 1 for row in ledger.values()),
            'valid_outputs': len(valid),
            'status_counts': dict(status_counts),
            'missing_input_ids': [key for key in ledger if not media[key]],
            'duplicate_inputs': {key: paths for key, paths in media.items() if len(paths) > 1},
            'missing_output_ids': [key for key in ledger if key not in valid],
            'invalid_outputs': invalid,
            'duplicate_output_ids': duplicates,
            'videos': ledger,
        }
        atomic_json(report, audit)
        return audit

    checkpoint()
    if not check_only:
        outputs.mkdir(parents=True, exist_ok=True)
        for video_id, row in ledger.items():
            if row['status'] != 'pending':
                continue
            source = videos / row['input_paths'][0]
            target = outputs / row['expected_output_filename']
            started = monotonic()
            try:
                subprocess.run(
                    ['bash', str(ROOT / 'run.sh'), str(source.resolve()), str(target.resolve()),
                     '--team-id', team_id, '--device', 'cpu'],
                    env={**os.environ, 'EGOSIGHT_PYTHON': sys.executable}, check=True,
                )
                valid, invalid, duplicates = inspect_outputs(outputs, kit, team_id)
                if video_id not in valid:
                    raise ValueError(invalid.get(target.name, 'Inference did not write a valid output'))
                row.update(valid[video_id], status='processed')
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                row.update(status='failed', error=str(exc).replace(str(videos.resolve()), '<videos>')
                           .replace(str(outputs.resolve()), '<outputs>').replace(str(ROOT), '<track4>'))
            row['inference_seconds'] = round(monotonic() - started, 3)
            checkpoint()
            print(f'{len(valid)} / {len(ledger)} | {video_id} | {row["status"]}', flush=True)
    return checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('videos', type=Path, help='Directory containing <video_id>.mp4 files')
    parser.add_argument('outputs', type=Path)
    parser.add_argument('--team-id', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true', help='Audit without running inference')
    args = parser.parse_args()
    if not args.videos.is_dir():
        parser.error('Input directory does not exist')
    try:
        audit = run_manifest(args.videos, args.outputs, args.report, args.team_id, args.check_only)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'{audit["valid_outputs"]} / {audit["expected"]} locally valid outputs')
    return int(bool(audit['missing_output_ids'] or audit['invalid_outputs']
                    or audit['duplicate_output_ids'] or audit['duplicate_inputs']))


if __name__ == '__main__':
    raise SystemExit(main())
