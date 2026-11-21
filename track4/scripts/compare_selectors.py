"""Compare selection algorithms using saved real candidate scores.

Input JSON: duration, candidates (start,end,confidence,visual,verb,noun),
and optional ground_truth (actions, highlights). This development format is
NOT the organizer's unknown annotation schema. No metrics without real GT.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from dentilens.selection import select
from dentilens.schema import atomic_json


def measure(highlights,gt,duration):
    # Time-domain F1, equivalent to an infinitely fine frame grid for CFR media.
    # Reference highlights must already represent consensus top-15% frames.
    reference=gt['highlights']
    for segments in (highlights,reference):
        previous=0
        for s in segments:
            if not previous<=s['start']<s['end']<=duration:
                raise ValueError('GT/highlight intervals must be chronological, disjoint and in bounds')
            previous=s['end']
    overlap=sum(max(0,min(a['end'],b['end'])-max(a['start'],b['start'])) for a in highlights for b in reference)
    denom=sum(s['end']-s['start'] for s in highlights+reference)
    pairs={(a['verb'],a['noun']) for a in gt['actions']}
    captured={(a['verb'],a['noun']) for a in gt['actions'] if any(
        min(a['end'],h['end'])>max(a['start'],h['start']) for h in highlights)}
    return dict(time_domain_highlight_f1=2*overlap/denom if denom else None,
                distinct_gt_pair_coverage_proxy=len(captured)/len(pairs) if pairs else None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidates',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    data=json.loads(args.candidates.read_text())
    results={}
    for strategy in ('uniform','motion','confidence','diversity'):
        selected=select(data['candidates'],data['duration'],strategy=strategy)
        results[strategy]=dict(highlights=selected,selected_seconds=sum(h['end']-h['start'] for h in selected))
        if 'ground_truth' in data:
            results[strategy]['metrics']=measure(selected,data['ground_truth'],data['duration'])
    atomic_json(args.output,dict(results=results,official_evaluation=False,
        note='Development proxy metrics only; no official action mAP implementation or organizer GT adapter.'))
