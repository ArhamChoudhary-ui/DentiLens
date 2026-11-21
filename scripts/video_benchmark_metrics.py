"""Frame localization, identity association and duration metrics."""
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluate_video_instruments import iou


def matched_objects(truth, predictions):
    if not truth or not predictions:
        return []
    overlaps=np.array([[iou(a['box'],b['box']) for b in predictions] for a in truth])
    valid=overlaps>=.5
    weights=np.where(valid, min(len(truth),len(predictions))+overlaps,0.)
    rows,cols=linear_sum_assignment(weights,maximize=True)
    return [(i,j) for i,j in zip(rows,cols) if valid[i,j]]


def track_metrics(annotation,predictions,video):
    gt=defaultdict(list)
    observations=defaultdict(list)
    review_frames=set(annotation['frames'])
    for row in annotation['objects']:
        if row['box'] is not None:
            gt[row['frame']].append(row)
    for row in predictions:
        if row['frame'] not in review_frames:
            raise ValueError('A prediction falls outside the reviewed cadence.')
        if not row.get('duplicate_of'):
            observations[row['frame']].append(row)
    pairs=Counter()
    previous={}
    unmatched_track_ids=set()
    matched_track_ids=set()
    matched_truth=set()
    details={}
    switches=fragmentation=0
    detect_seconds=correct_seconds=incorrect_seconds=visible_seconds=0.
    for index,frame in enumerate(annotation['frames']):
        boundary=(annotation['frames'][index+1] if index+1<len(annotation['frames']) else video['frames'])
        time=frame/video['fps']
        duration=(boundary-frame)/video['fps']
        truth=gt[frame]
        predictions=observations[frame]
        assignments=dict(matched_objects(truth,predictions))
        hit_predictions=set(assignments.values())
        visible_ids={row['track_id'] for row in truth}
        previous={key:value for key,value in previous.items() if key in visible_ids}
        for i,row in enumerate(truth):
            key=row['track_id']
            detail=details.setdefault(key,dict(first=time,first_detection=None,first_correct=None,
                                              visible=0.,detected=0.,correct=0.,incorrect=0.,named_seconds=Counter(),prediction_ids=set()))
            visible_seconds+=duration
            detail['visible']+=duration
            if i not in assignments:
                if key in previous:
                    previous[key]=(previous[key][0],False)
                continue
            detection=predictions[assignments[i]]
            track=detection.get('track_id')
            matched_truth.add(key)
            detect_seconds+=duration
            detail['detected']+=duration
            if detail['first_detection'] is None:
                detail['first_detection']=time
            if track:
                pairs[key,track]+=1
                matched_track_ids.add(track)
                detail['prediction_ids'].add(track)
                if key in previous:
                    old,was_matched=previous[key]
                    switches+=int(old!=track)
                    fragmentation+=int(not was_matched)
                previous[key]=(track,True)
            name=detection.get('track_identity')
            if row['visibility']!='ambiguous' and row['label']!='unknown' and name and name.get('accepted'):
                detail['named_seconds'][name['class_id']]+=duration
                if name['class_id']==row['label']:
                    correct_seconds+=duration
                    detail['correct']+=duration
                    if detail['first_correct'] is None:
                        detail['first_correct']=time
                else:
                    incorrect_seconds+=duration
                    detail['incorrect']+=duration
        unmatched_track_ids.update(predictions[j]['track_id'] for j in range(len(predictions))
                                   if j not in hit_predictions and predictions[j].get('track_id'))
    g_ids=sorted({key for key,_ in pairs})
    p_ids=sorted({key for _,key in pairs})
    idtp=0
    if pairs:
        table=np.array([[pairs[a,b] for b in p_ids] for a in g_ids])
        rows,columns=linear_sum_assignment(table,maximize=True)
        idtp=int(table[rows,columns].sum())
    n_truth=sum(len(values) for values in gt.values())
    n_preds=sum(len(values) for values in observations.values())
    for detail in details.values():
        detail['prediction_ids']=sorted(detail['prediction_ids'])
        named=detail['named_seconds']
        detail['name_consistency']=max(named.values())/sum(named.values()) if named else None
        for field in ('first_detection','first_correct'):
            detail[field+'_delay']=detail[field]-detail['first'] if detail[field] is not None else None
    return dict(idf1=2*idtp/(n_truth+n_preds) if n_truth+n_preds else None,
                idtp=idtp, ground_truth_detections=n_truth, predicted_detections=n_preds,
                identity_switches=switches, fragmentations=fragmentation,
                false_tracks=len(unmatched_track_ids-matched_track_ids),
                ground_truth_tracks_with_multiple_prediction_ids=sum(len(r['prediction_ids'])>1 for r in details.values()),
                visible_seconds=visible_seconds,missed_visible_seconds=visible_seconds-detect_seconds,
                correctly_named_visible_seconds=correct_seconds,incorrectly_named_visible_seconds=incorrect_seconds,matched_truth_tracks=len(matched_truth),
                details=details)


def stratified_detection(samples, cutoff, width, height):
    from evaluate_detector import detection_metrics
    counts=Counter()
    hits=Counter()
    for truth,predictions,_ in samples:
        eligible=[dict(box=box) for score,box in predictions if score>=cutoff]
        matched={i for i,_ in matched_objects(truth,eligible)}
        for i,row in enumerate(truth):
            box=row['box']
            labels=[row['visibility']]
            if min((box[2]-box[0])*width,(box[3]-box[1])*height)<16:
                labels.append('short_edge_under_16px')
            for label in labels:
                counts[label]+=1
                hits[label]+=int(i in matched)
    result=detection_metrics(samples,cutoff)
    result['strata']={label:dict(matched=hits[label],total=n,recall=hits[label]/n)
                      for label,n in sorted(counts.items())}
    return result


def annotation_disagreements(primary, second):
    if (primary['recording'],primary['frames'])!=(second['recording'],second['frames']):
        raise ValueError('Annotation reviews must use the same recording and cadence.')
    if primary['review']['reviewer']==second['review']['reviewer']:
        raise ValueError('Choose a different reviewer for the independent annotation.')
    grouped = []
    for annotation in (primary,second):
        samples=defaultdict(list)
        for row in annotation['objects']:
            if row['box'] is not None:
                samples[row['frame']].append(row)
        grouped.append(samples)
    cases=[]
    correspondences=Counter()
    for frame in primary['frames']:
        left,right=(group[frame] for group in grouped)
        matches=matched_objects(left,right)
        for index,other in matches:
            a,b=left[index],right[other]
            correspondences[a['track_id'],b['track_id']]+=1
            fields=[field for field in ('label','visibility','occluded') if a.get(field)!=b.get(field)]
            if fields:
                cases.append(dict(frame=frame,reason='class_or_visibility',fields=fields,primary=a,second=b))
            overlap=iou(a['box'],b['box'])
            if overlap<.75:
                cases.append(dict(frame=frame,reason='box_geometry',iou=overlap,primary=a,second=b))
        for side,values,matched in (('primary',left,{i for i,_ in matches}),('second',right,{j for _,j in matches})):
            for i,item in enumerate(values):
                if i not in matched:
                    cases.append(dict(frame=frame,reason='unmatched_box',**{side:item}))
    left_events={row['track_id']:row for row in primary.get('events',[])}
    right_events={row['track_id']:row for row in second.get('events',[])}
    for (a,b),count in sorted(correspondences.items()):
        if sum(key[0]==a for key in correspondences)>1 or sum(key[1]==b for key in correspondences)>1:
            cases.append(dict(reason='track_split_or_merge',primary_track=a,second_track=b,matched_samples=count))
        left,right=left_events.get(a),right_events.get(b)
        if left and right and any(left[field]!=right[field] for field in ('entry','exit','last_seen')):
            cases.append(dict(reason='track_timing',primary=left,second=right))
    return dict(recording=primary['recording'],reviewers=[primary['review']['reviewer'],second['review']['reviewer']],
                reviewed_samples=len(primary['frames']),disagreements=cases)
