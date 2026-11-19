"""Download the published AlphaDent checkpoint and verify the recorded hash."""
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


def main():
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / 'reference/model-source.json').read_text())
    target = root / 'models/alphadent.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with target.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() == provenance['sha256']:
                print('The verified checkpoint is already installed.')
                return
        raise ValueError('An existing checkpoint has a different hash. Move it before downloading.')
    staged = None
    try:
        digest = hashlib.sha256()
        downloaded = 0
        with urllib.request.urlopen(provenance['source'], timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.download', delete=False) as stream:
                staged = Path(stream.name)
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
        if downloaded != provenance['bytes'] or digest.hexdigest() != provenance['sha256']:
            raise ValueError('The downloaded checkpoint failed verification')
        staged.replace(target)
        print(f'Installed verified checkpoint at {target}')
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
