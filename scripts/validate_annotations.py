"""Validate frame annotations, taxonomy, license and crop-parent relationships."""
import argparse

from instrument_dataset import Path, load_dataset, validate


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    args = parser.parse_args()
    metadata, frames, crops = load_dataset(args.dataset)
    validate(metadata, frames, crops)
    print(f'Validated {len(frames)} frames and {len(crops)} crops against {metadata["version"]}.')
