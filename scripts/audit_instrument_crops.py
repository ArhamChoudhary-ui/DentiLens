"""Audit saved crop evidence and replay local holdout recordings."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from instrument_identity import DEFAULT_CROP_PADDING, classify_pipeline, context_model, crop_geometry, describe_tracks, identity_pipeline
from instrument_video import measure_instruments


def audit(rows, video, predictions, padding=DEFAULT_CROP_PADDING):
    for region in rows:
        region['identity'] = predictions.get(region['detection_id'])
    tracks, counts = describe_tracks(rows,video,[])
    evidence = []
    for row in rows:
        identity = row['identity'] or {}
        alternatives = identity.get('alternatives',[])
        evidence.append(dict(detection_id=row['detection_id'],frame=row['frame'],time=row['time'],box=row['box'],
            detection_confidence=row['confidence'],canonical_id=row.get('duplicate_of') or row['detection_id'],
            track_id=row.get('track_id'),status=row['instrument_status'],reason=row['reason_code'],
            crop=crop_geometry(row['box'],video['width'],video['height'],padding),model=identity.get('model'),
            top_classes=alternatives))
    uncertain = [entry for entry in evidence if entry['status'] in ('type_uncertain','track_interrupted','unclassified_candidate')]
    counts['uncertain_encoded_short_side'] = dict(Counter(
        'under 12 px' if side<12 else '12–23 px' if side<24 else '24 px or more'
        for entry in uncertain for side in [min(entry['crop']['encoded_width'],entry['crop']['encoded_height'])]))
    return dict(diagnostics=counts,tracks=tracks,evidence=evidence)


def read_clip(source):
    capture=cv2.VideoCapture(str(source))
    try:
        fps=capture.get(cv2.CAP_PROP_FPS)
        frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or not frames:
            raise ValueError(f'Could not read {source.name}.')
        return dict(fps=fps,frames=frames,duration=frames/fps,width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exam',type=Path,required=True)
    parser.add_argument('--clip',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    opt=parser.parse_args()
    torch.set_num_threads(4)
    root=ROOT/'models'
    identities=json.loads((opt.exam/'instrument-identities.json').read_text())['predictions']
    source=opt.exam/'source.mp4'
    rows=json.loads((opt.exam/'instruments-video.json').read_text())['observations']
    output={'saved_exam':audit(rows,read_clip(source),identities),'holdouts':[],'padding_trials':{}}
    models=identity_pipeline(root)
    ctx=context_model(root)
    # Keep playback geometry fixed and test only the instrument crop padding.
    for padding in (.04,.10,.20):
        scoped=[(model,{**info,'crop_padding':padding}) for model,info in models]
        predictions=classify_pipeline(source,rows,scoped,ctx)
        report=audit(rows,read_clip(source),predictions,padding)
        by_id={entry['detection_id']:entry['status'] for entry in report['evidence']}
        output['padding_trials'][str(padding)]=dict(diagnostics=report['diagnostics'],probe_frames={
            item['detection_id']:by_id[item['detection_id']] for item in rows if 2.19<=item['time']<=2.61})
    for clip in opt.clip:
        with TemporaryDirectory() as temp:
            local=Path(temp)/'source.mp4'
            local.symlink_to(clip.resolve())
            video=read_clip(local)
            raw=measure_instruments(local,video,root,experimental=True)
            predictions=classify_pipeline(local,raw['observations'],models,ctx)
        output['holdouts'].append(dict(source=clip.name,sampled_frames=raw['samples'],
                                       **audit(raw['observations'],video,predictions)))
    opt.output.write_text(json.dumps(output,indent=2)+'\n')
    print(output['saved_exam']['diagnostics'])
    for bound,entry in output['padding_trials'].items():
        print('padding',bound,entry['diagnostics'])
    for entry in output['holdouts']:
        print(entry['source'],entry['sampled_frames'],entry['diagnostics'])
