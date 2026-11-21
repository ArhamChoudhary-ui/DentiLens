"""Build an offline archive after checking required assets and output contracts."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZIP_STORED,ZIP_DEFLATED,ZipFile

from check_submission import StudentKit,check_files


def verify_weights(root):
    expected=json.loads((root/'weights/receipt.json').read_text())
    for name,digest in expected.items():
        with (root/'weights'/name).open('rb') as stream:
            if file_digest(stream,'sha256').hexdigest()!=digest:
                raise ValueError(f'Incorrect model asset: {name}')
    return ['receipt.json',*expected]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outputs',type=Path)
    parser.add_argument('zipfile',type=Path)
    parser.add_argument('--team-id',required=True)
    parser.add_argument('--development',action='store_true')
    parser.add_argument('--organizer-validator',type=Path)
    args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+',args.team_id):
        parser.error('Team ID must be a safe folder name.')
    root=Path(__file__).resolve().parents[1]
    if not args.development and (not args.organizer_validator or not args.organizer_validator.is_file()):
        parser.error('Final packaging requires the actual organizer validator.')
    if args.team_id=='DEVELOPMENT' and not args.development:
        parser.error('Set the registered team ID for a final archive.')
    files=sorted(args.outputs.glob('*.json'))
    result=check_files(files,StudentKit(),team_id=args.team_id)
    if not files or result['errors'] or (not args.development and result['missing_video_ids']):
        parser.error(f'Output check failed: {result["errors"]}; missing {len(result["missing_video_ids"])} videos.')
    if not args.development:
        subprocess.run([sys.executable,str(args.organizer_validator.resolve()),'--dir',str(args.outputs.resolve())],
                       check=True)
    weights=verify_weights(root)
    folder=f'{args.team_id}-development' if args.development else args.team_id
    with ZipFile(args.zipfile,'x',compression=ZIP_DEFLATED) as archive:
        for name in ('README.md','requirements.txt','run.sh','Dockerfile'):
            archive.write(root/name,f'{folder}/{name}')
        for name in ('code','reference'):
            for path in sorted((root/name).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    archive.write(path,f'{folder}/{path.relative_to(root)}')
        for name in ('check_submission.py','evaluate_local.py','evaluate_fpv_local.py','run_manifest.py'):
            archive.write(root/'scripts'/name,f'{folder}/scripts/{name}')
        archive.write(root/"weights/README.md",f"{folder}/weights/README.md")
        for name in weights:
            archive.write(root/'weights'/name,f'{folder}/weights/{name}',
                          compress_type=ZIP_STORED if name.endswith(('.pth','.ckpt')) else ZIP_DEFLATED)
        for path in files:
            archive.write(path,f'{folder}/outputs/{path.name}')
        if not args.development:
            archive.writestr(f'{folder}/team_id.txt',args.team_id+'\n')
        else:
            message='Development sample. Not for organizer upload. Use --team-id DEVELOPMENT to reproduce sample output.\n'
            archive.writestr(f'{folder}/DEVELOPMENT_ONLY.txt',message)
    print(f'Wrote {args.zipfile} with {len(files)} locally valid output files.')


if __name__=='__main__':
    main()
