"""Run available manifest videos using the normal single-video CLI entry point."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from dentilens.contract import StudentKit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('videos', type=Path, help='Directory containing <video_id>.mp4 files')
    parser.add_argument('outputs', type=Path)
    parser.add_argument('--team-id', required=True)
    parser.add_argument('--mode', choices=['model', 'motion-baseline'], default='model')
    parser.add_argument('--check-only', action='store_true', help='List missing media without running inference')
    args = parser.parse_args()
    kit = StudentKit()
    media = {}
    for path in sorted(args.videos.rglob('*.mp4')):
        if path.stem in media:
            parser.error(f'Duplicate input ID: {path.stem}')
        media[path.stem] = path.resolve()
    missing = sorted(set(kit.entries) - set(media))
    print(json.dumps({'expected': len(kit.entries), 'available': len(set(kit.entries) & set(media)),
                      'missing': missing}, indent=2), flush=True)
    if args.check_only:
        return int(bool(missing))
    if missing:
        parser.error('Download the listed source videos first. No placeholder outputs are generated.')
    args.outputs.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, 'DENTILENS_PYTHON': sys.executable}
    for video_id in kit.entries:
        subprocess.run(['bash', str(ROOT / 'run.sh'), str(media[video_id]),
                        str(args.outputs.resolve() / (video_id + '.json')),
                        '--team-id', args.team_id, '--mode', args.mode], env=environment, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
