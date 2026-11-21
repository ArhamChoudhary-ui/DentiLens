"""Summarize reproducible resolution runs and classify unmatched proposal evidence."""
import argparse
from collections import Counter
from pathlib import Path

from instrument_dataset import load_dataset, read_json, write_json
from evaluate_video_instruments import iou


def summarize(frames, metrics, errors):
    frames_by_id={row['id']:row for row in frames}
    misses=Counter()
    total_small=0
    for row in frames:
        if row['recording'] not in metrics['frames_by_recording'] or row['split']!='valid':
            continue
        for item in row['annotations']:
            if item['label']!='non_instrument' and small(item['box'],row):
                total_small+=1
    categories=Counter()
    by_context=Counter()
    for case in errors:
        source=frames_by_id[case['id']]
        if case['kind']=='missed':
            if small(case['box'],source):
                misses['small']+=1
            if case['best_proposal_iou']>=.5:
                misses['had_overlap_in_ranked_proposals']+=1
            continue
        if case['best_annotated_iou']>=.5:
            categories['overlap_with_already_matched_tool']+=1
            continue
        context=[a for a in source['annotations'] if a['label']=='non_instrument']
        match=max(context,key=lambda a:iou(a['box'],case['box']),default=None)
        if match and iou(match['box'],case['box'])>=.5:
            categories['overlap_with_annotated_non_instrument']+=1
            by_context[match['original_class']]+=1
        elif case['best_annotated_iou']>=.1:
            categories['partial_tool_overlap']+=1
        else:
            categories['low_tool_overlap_unverified']+=1
    return dict(metrics=metrics,unmatched_categories=dict(categories),context_counts=dict(by_context),
                missed_with_ranked_overlap=misses['had_overlap_in_ranked_proposals'],
                small_boxes=total_small,small_matched=total_small-misses['small'])


def small(box,row):
    return min((box[2]-box[0])*row['width'],(box[3]-box[1])*row['height'])<16


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('runs',type=Path,help='Directory containing <size>.json and <size>-errors.json.')
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    _,frames,_=load_dataset(args.dataset)
    runs={}
    for size in (512,640,768):
        file=args.runs/f'{size}.json'
        result=read_json(file)
        if result.get('split')!='valid' or result.get('max_per_recording') is not None:
            raise ValueError('Comparison requires the full shared validation split.')
        runs[str(size)]=summarize(frames,result,read_json(args.runs/f'{size}-errors.json'))
    cohorts={(tuple(sorted(row['metrics']['frames_by_recording'].items())),row['metrics']['checkpoint_sha256'],
              row['metrics']['archive_sha256'],row['metrics']['score_cutoff']) for row in runs.values()}
    if len(cohorts)!=1:
        raise ValueError('Resolution runs must use identical data, weights and operating threshold.')
    write_json(args.output,runs)
