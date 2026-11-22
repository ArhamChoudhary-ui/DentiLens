"""Explicit local FPVSum diagnostic against public continuous frame scores."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from egosight.contract import StudentKit,validate_submission


def highlight_f1(scores, highlights, duration):
    scores=np.asarray(scores,dtype=float)
    n=scores.size
    if scores.ndim!=1 or n==0 or not np.isfinite(scores).all():
        raise ValueError('Expected a finite one-dimensional human score array.')
    top=np.argsort(-scores,kind='stable')[:max(1,int(np.floor(.15*n)))]
    truth=np.zeros(n,dtype=bool); truth[top]=True
    # Mapping assumes uniform annotation sampling over the matched full duration.
    time=np.arange(n)*duration/n
    predicted=np.zeros(n,dtype=bool)
    for item in highlights:
        start,end=item['segment']
        predicted|=(time>=start)&(time<end)
    tp=int((truth&predicted).sum())
    total=int(truth.sum()+predicted.sum())
    return dict(f1=2*tp/total if total else None,true_positive=tp,target_frames=int(truth.sum()),
                predicted_frames=int(predicted.sum()))


def main():
    from scipy.io import loadmat

    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('annotations',type=Path)
    p.add_argument('predictions',type=Path,nargs='+')
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    kit=StudentKit()
    annotation={row['Name']:row for row in loadmat(args.annotations,simplify_cells=True)['gt_data']}
    report={}
    confusion={}
    per_class=Counter()
    correct=Counter()
    for path in args.predictions:
        prediction=json.loads(path.read_text())
        validate_submission(prediction,kit,filename=path.name)
        name=prediction['video_id']
        if name in report or kit.entries[name].dataset!='fpvsum' or name not in annotation:
            raise ValueError(f'Requires a unique FPVSum video with public ground truth: {name}')
        expected=annotation[name]['Name'].split('_',1)[0].lower()
        expected={'skiing':'ski'}.get(expected,expected)
        if expected not in kit.labels:
            raise ValueError('Public annotation category is not in the organizer vocabulary.')
        predicted=prediction['label']['name']
        per_class[expected]+=1
        correct[expected]+=int(predicted==expected)
        confusion.setdefault(expected,Counter())[predicted]+=1
        report[name]=dict(ground_truth=expected,prediction=predicted,
                          **highlight_f1(annotation[name]['Score'],prediction['highlights'],prediction['duration_sec']))
    payload=dict(scope='Local FPVSum diagnostics; uniform annotation grid and deterministic top-15% tie ordering.',
                 videos=report,mean_highlight_f1=float(np.mean([row['f1'] for row in report.values()])),
                 label_accuracy=sum(correct.values())/len(report),confusion_matrix=confusion,
                 per_class_accuracy={key:correct[key]/value for key,value in per_class.items()})
    args.report.write_text(json.dumps(payload,indent=2)+'\n')
    print(payload)


if __name__=='__main__':
    main()
