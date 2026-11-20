"""Download the pinned official YOLOv8n training backbone with integrity checks."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, default=root / 'models/yolov8n.pt')
    args = parser.parse_args()
    metadata = json.loads((root / 'reference/tooth-backbone.json').read_text())
    args.target.parent.mkdir(parents=True, exist_ok=True)
    if args.target.exists():
        with args.target.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != metadata['sha256']:
            raise ValueError('Existing backbone hash differs from the pinned release')
        print('Verified training backbone is already installed.')
    else:
        with tempfile.TemporaryDirectory(dir=args.target.parent) as temporary:
            staged = Path(temporary) / 'backbone.download'
            with urllib.request.urlopen(metadata['url'], timeout=60) as response, staged.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            with staged.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if staged.stat().st_size != metadata['bytes'] or digest != metadata['sha256']:
                raise ValueError('Downloaded training backbone failed verification')
            staged.replace(args.target)
        print(f'Installed verified training backbone at {args.target}')
