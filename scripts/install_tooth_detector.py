"""Copy a completed, hash-verified training checkpoint into the local app."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile


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
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
    from dental_modes import TRAINED_MODES

    classes = tuple(provenance['dataset']['output_classes'])
    checkpoint = next((name for name, _, labels in TRAINED_MODES.values() if labels == classes), None)
    if checkpoint is None:
        parser.error('Checkpoint classes do not match a review mode')
    destination = args.model_directory
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        staged = Path(temporary)
        shutil.copyfile(source, staged / 'weights')
        (staged / 'provenance').write_text(json.dumps(provenance, indent=2) + '\n')
        (staged / 'weights').replace(destination / f'{checkpoint}.pt')
        (staged / 'provenance').replace(destination / f'{checkpoint}.json')
    print(f'Installed {checkpoint} in {destination}')
