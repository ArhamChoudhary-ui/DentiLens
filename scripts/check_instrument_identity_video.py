"""Run detection and crop classification on whole local videos."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from instrument_identity import classify_observations, identity_model
from instrument_video import measure_instruments


def check(source, model, info):
    cap = cv2.VideoCapture(str(source))
    try:
        video = {'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 'fps':cap.get(cv2.CAP_PROP_FPS)}
    finally:
        cap.release()
    if not video['frames'] or not video['fps']:
        raise ValueError(f'Could not open {source}.')
    # The detector creates temporary tracker configuration by the source.
    with tempfile.TemporaryDirectory() as folder:
        local = Path(folder)/'source.mp4'
        local.symlink_to(source.resolve())
        scan = measure_instruments(local,video,ROOT/'models',experimental=True)
        identities = classify_observations(local,scan['observations'],model,info)
    accepted = Counter(row['label'] for row in identities.values() if row and row['accepted'])
    return {'source':source.name,'sampled_frames':scan['samples'],'proposal_boxes':len(identities),
            'named_boxes':dict(accepted),'uncertain_or_unclassified':len(identities)-sum(accepted.values())}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('videos',type=Path,nargs='+')
    options=parser.parse_args()
    model, info = identity_model(ROOT/'models')
    print(json.dumps([check(source,model,info) for source in options.videos],indent=2))
