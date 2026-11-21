"""Fail on recording, patient, exact-image or flagged perceptual split overlap."""
import argparse
import json

from instrument_dataset import Path, leakage, load_dataset


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    args = parser.parse_args()
    _, frames, _ = load_dataset(args.dataset)
    results = leakage(frames)
    print(json.dumps(results,indent=2))
    raise SystemExit(any(value for value in results.values() if isinstance(value,list)))
