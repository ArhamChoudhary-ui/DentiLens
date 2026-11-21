"""Public-label development metrics. This is separate from organizer scoring."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from dentilens.contract import StudentKit, validate_submission


def seconds(value):
    hour,minute,second=value.split(':')
    return 3600*float(hour)+60*float(minute)+float(second)


def overlap(a,b):
    return max(0.,min(a[1],b[1])-max(a[0],b[0]))


def average_precision(hits,count):
    precision=[0.]
    recall=[0.]
    correct=0
    for index,hit in enumerate(hits,1):
        correct+=hit
        precision.append(correct/index)
        recall.append(correct/count)
    precision.append(0.)
    recall.append(1.)
    for i in range(len(precision)-2,-1,-1):
        precision[i]=max(precision[i],precision[i+1])
    return sum((recall[i]-recall[i-1])*precision[i] for i in range(1,len(recall)))


def epic_scores(outputs,ground_truth):
    gt=defaultdict(lambda:defaultdict(list))
    pred=defaultdict(list)
    coverage={}
    for video,output in outputs.items():
        visible=set(); pairs=set()
        for item in ground_truth[video]:
            pair=(item['verb'],item['noun'])
            span=item['segment']
            gt[pair][video].append(span)
            pairs.add(pair)
            if sum(overlap(span,h['segment']) for h in output['highlights']) >= (span[1]-span[0])/2:
                visible.add(pair)
        coverage[video]=dict(covered_pairs=len(visible),reference_pairs=len(pairs))
        for item in output['actions']:
            pred[item['verb'],item['noun']].append((item['score'],video,item['segment']))
    ap={}
    for threshold in (.1,.2,.3,.4,.5):
        scores=[]
        for pair,videos in gt.items():
            remaining={key:list(spans) for key,spans in videos.items()}
            hits=[]
            for _,video,candidate in sorted(pred[pair],key=lambda a:-a[0]):
                matches=remaining.get(video,[])
                ratios=[overlap(span,candidate)/(span[1]-span[0]+candidate[1]-candidate[0]-overlap(span,candidate))
                        for span in matches]
                best=max(range(len(matches)),key=ratios.__getitem__,default=None)
                hit=best is not None and ratios[best]>=threshold
                hits.append(hit)
                if hit:
                    matches.pop(best)
            scores.append(average_precision(hits,sum(map(len,videos.values()))))
        ap[str(threshold)]=sum(scores)/len(scores) if scores else None
    return dict(local_action_map=ap,local_coverage_at_half_action=coverage)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations',type=Path)
    parser.add_argument('predictions',type=Path,nargs='+')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    kit=StudentKit()
    outputs={}
    ground_truth=defaultdict(list)
    for path in args.predictions:
        output=json.loads(path.read_text())
        validate_submission(output,kit,filename=path.name)
        identifier=output['video_id']
        if identifier in outputs or kit.entries[identifier].dataset!='epic_kitchens_100':
            raise ValueError('Use unique EPIC-KITCHENS validation outputs.')
        outputs[identifier]=output
    verbs=kit.verbs
    nouns=kit.nouns
    with args.annotations.open(newline='') as stream:
        for row in csv.DictReader(stream):
            video=row['video_id']
            if video in outputs and row['verb'] in verbs and row['noun'] in nouns:
                start,stop=seconds(row['start_timestamp']),seconds(row['stop_timestamp'])
                if stop>start:
                    ground_truth[video].append(dict(verb=row['verb'],noun=row['noun'],segment=[start,stop]))
    if set(ground_truth)!=set(outputs):
        raise ValueError('Missing public ground truth for one or more outputs.')
    result=dict(scope='Public EPIC subset development metrics. Coverage uses >=50% of action duration.',
                videos={video:len(rows) for video,rows in ground_truth.items()},**epic_scores(outputs,ground_truth))
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':
    main()
