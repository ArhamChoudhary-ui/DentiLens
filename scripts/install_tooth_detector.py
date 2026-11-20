"""Copy a completed, hash-verified training checkpoint into the local app."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('training_directory', type=Path)
    parser.add_argument('--model-directory', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'models')
    args = parser.parse_args()
    provenance = json.loads((args.training_directory / 'provenance.json').read_text())
    source = args.training_directory / 'weights/best.pt'
    digest = hashlib.file_digest(source.open('rb'), 'sha256').hexdigest()
    if digest != provenance['weights_sha256']:
        parser.error('Training checkpoint hash does not match provenance')
    destination = args.model_directory
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination / 'tooth-detector.pt')
    (destination / 'tooth-detector.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Installed tooth detector in {destination}')
