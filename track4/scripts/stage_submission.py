"""Build a Track 4 staging archive with the organizer's six root entries."""
import argparse
from pathlib import Path
from shutil import copy2, copytree, ignore_patterns
from zipfile import ZIP_DEFLATED, ZipFile

from check_submission import StudentKit, check_files

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = ('run.sh', 'code', 'weights', 'requirements.txt', 'outputs', 'README.md')


def stage(outputs, destination, team_id):
    files = sorted(outputs.glob('*.json'))
    audit = check_files(files, StudentKit(), team_id=team_id)
    if not files or audit['errors']:
        raise ValueError(f'Invalid staging outputs: {audit["errors"]}')
    package = destination / 'Rajasthani_Staging'
    package.mkdir(parents=True)
    launcher = (ROOT / 'run.sh').read_text().replace(
        'set -euo pipefail\n',
        'set -euo pipefail\n'
        'if [[ "$#" -ne 2 ]]; then\n'
        '  printf \'Usage: %s <video.mp4> <out.json>\\n\' "$0" >&2\n'
        '  exit 2\n'
        'fi\n',
        1,
    ).replace('exec "$PYTHON"', 'if ! command -v "$PYTHON" >/dev/null 2>&1; then\n'
              '  printf \'Install Python 3.11 and the packages in requirements.txt.\\n\' >&2\n'
              '  exit 127\nfi\nexec "$PYTHON"')
    entry = package / 'run.sh'
    entry.write_text(launcher)
    entry.chmod(0o755)

    code = package / 'code'
    copytree(ROOT / 'code', code, ignore=ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
    kit = code / 'reference' / 'student_package'
    kit.mkdir(parents=True)
    for name in ('manifest.csv', 'supported_classes.json', 'TEAM_README.md'):
        copy2(ROOT / 'reference' / 'student_package' / name, kit / name)
    copy2(ROOT / 'scripts' / 'check_submission.py', code / 'check_submission.py')
    (code / 'team_id.txt').write_text(team_id + '\n')

    weights = package / 'weights'
    weights.mkdir()
    for name in ('README.md', 'receipt.json', 'epic-classes.json'):
        copy2(ROOT / 'weights' / name, weights / name)
    copy2(ROOT / 'requirements.txt', package / 'requirements.txt')

    results = package / 'outputs'
    results.mkdir()
    for path in files:
        copy2(path, results / path.name)

    readme = (ROOT / 'README.md').read_text().split('## Organizer kit and batch validation')[0]
    first, body = readme.split('\n', 1)
    body = body.replace('**8 of 187 manifest videos**',
                        f'**{audit["valid_videos"]} of 187 manifest videos**').replace(
                            '**179 source videos**', f'**{len(audit["missing_video_ids"])} prediction files**'
                        ).replace('were unavailable in the supplied\nworkspace', 'are outstanding').replace(
                            'scripts/check_submission.py', 'code/check_submission.py'
                        ).replace(
                            "export DENTILENS_TEAM_ID='your-registered-team-id'\n", ''
                        ).replace(
                            '[Actual EPIC and FPV JSON samples](reference/samples)',
                            '[Locally checked predictions](outputs)'
                        )
    (package / 'README.md').write_text(
        first + '\n\n'
        'This staging package uses the local run identifier stored in `code/team_id.txt`. '
        'After confirming the Samsung TEAM_ID, use it for the archive root and regenerate '
        'all outputs with that ID. Keep the presentation and demo separate.\n\n'
        'Download both upstream checkpoints using `weights/README.md`. They are '
        'excluded from the shareable staging ZIP while redistribution terms are checked. '
        'Place them in `weights/`. Inference verifies their hashes.\n' + body +
        '\n## Validate the output directory\n\n'
        '` .venv/bin/python code/check_submission.py --dir outputs `. '
        'The local checker returns a nonzero status for incomplete manifest coverage. '
        'The supplied kit lacks the official validator.\n'
    )
    archive = destination / 'Rajasthani_Track4_Staging.zip'
    with ZipFile(archive, 'x', compression=ZIP_DEFLATED) as bundle:
        for name in LAYOUT:
            path = package / name
            if path.is_file():
                bundle.write(path, f'{package.name}/{name}')
            else:
                for child in sorted(path.rglob('*')):
                    if child.is_file():
                        bundle.write(child, f'{package.name}/{child.relative_to(package)}')
    print(f'{archive} — {audit["valid_videos"]}/187 locally verified outputs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outputs', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--team-id', default='DEVELOPMENT')
    args = parser.parse_args()
    stage(args.outputs, args.destination, args.team_id)
