"""Compare trained classifiers on unchanged video detector proposals."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from instrument_identity import classify_observations, identity_model


def replay(exam, classifiers, frame_numbers):
    rows = json.loads((exam/'instruments-video.json').read_text())['observations']
    results = []
    for directory in classifiers:
        model, info = identity_model(directory.resolve())
        predictions = classify_observations(exam/'source.mp4',rows,model,info)
        named = Counter(entry['label'] for entry in predictions.values() if entry and entry['accepted'])
        focused = [
            {'time':row['time'],'frame':row['frame'],'detection_id':row['detection_id'],
             'prediction':predictions[row['detection_id']]}
            for row in rows if row['frame'] in frame_numbers
        ]
        results.append({'model':info['model'],'proposals':len(rows),'named_boxes':dict(named),'checked_frames':focused})
    return results


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exam',type=Path)
    parser.add_argument('classifiers',type=Path,nargs='+')
    parser.add_argument('--frames',type=int,nargs='*',default=[])
    options = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(replay(options.exam,options.classifiers,options.frames),indent=2))
