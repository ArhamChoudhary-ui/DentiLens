"""Install pinned detector and segmenter snapshots with weight integrity checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path

# Use the standard HTTPS downloader, which can resume partial downloads.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import snapshot_download


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=['dino', 'sam2'], help='Install only the selected component')
    args = parser.parse_args()
    manifest = json.loads((root / 'reference/grounded-models.json').read_text())
    for name, model in manifest.items():
        if args.only and name != args.only:
            continue
        destination = root / 'models/grounded' / name
        patterns = ['*.json', '*.txt', 'model.safetensors']
        weights = destination / 'model.safetensors'
        if weights.is_file():
            with weights.open('rb') as stream:
                existing_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
            if existing_hash != model['weights_sha256']:
                raise ValueError(f'Remove the corrupt checkpoint at {weights} before retrying')
            patterns.remove('model.safetensors')
        snapshot_download(model['repository'], revision=model['revision'],
                          local_dir=destination,
                          allow_patterns=patterns)
        with (destination / 'model.safetensors').open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != model['weights_sha256']:
            raise ValueError(f'Checkpoint integrity check failed for {name}')
        (destination / 'provenance.json').write_text(json.dumps(model, indent=2) + '\n')
        print(f'Verified {name} at revision {model["revision"]}', flush=True)


if __name__ == '__main__':
    main()
