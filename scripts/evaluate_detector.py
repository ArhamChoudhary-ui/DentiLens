"""Evaluate the installed class-agnostic video detector against labelled full frames."""
import argparse
from collections import Counter
from zipfile import ZipFile

import numpy as np
import torch
from ultralytics import YOLO

from evaluate_video_instruments import iou
from instrument_dataset import ROOT, Path, checksum, image_from_archive, load_dataset, read_json, validate, write_json


def ranked_matches(samples, overlap=.5, score_floor=0.):
    ranked = sorted(
        ((score, index, serial, box) for index, (_, predictions, _) in enumerate(samples)
         for serial, (score, box) in enumerate(predictions) if score >= score_floor),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    remaining = [list(range(len(truth))) for truth, _, _ in samples]
    for score, index, _, box in ranked:
        truth, _, negatives = samples[index]
        best = max(remaining[index], key=lambda item:iou(box, truth[item]['box']), default=None)
        if best is not None and iou(box, truth[best]['box']) >= overlap:
            remaining[index].remove(best)
            yield score,truth[best]['label'],None
        else:
            negative = max(negatives,key=lambda item:iou(box,item['box']),default=None)
            yield score,None,negative['original_class'] if negative and iou(box,negative['box'])>=overlap else None


def average_precision(samples, overlap=.5):
    positives = sum(len(truth) for truth,_,_ in samples)
    if not positives:
        return None
    correct = np.array([label is not None for _,label,_ in ranked_matches(samples,overlap)],dtype=int)
    if not correct.size:
        return 0.
    hits = correct.cumsum()
    precision = hits / np.arange(1,len(hits)+1)
    recall = hits / positives
    return float(np.mean([precision[recall >= threshold].max(initial=0.) for threshold in np.linspace(0,1,101)]))


def detection_metrics(samples, cutoff):
    truth = Counter(annotation['label'] for objects,_,_ in samples for annotation in objects)
    matches = list(ranked_matches(samples,score_floor=cutoff))
    matched = Counter(label for _,label,_ in matches if label is not None)
    tp = sum(matched.values())
    fp = len(matches)-tp
    fn = sum(truth.values())-tp
    ap = [average_precision(samples,iou_cutoff) for iou_cutoff in np.linspace(.5,.95,10)]
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None, ap50=ap[0],
                map50_95=float(np.mean(ap)) if ap[0] is not None else None,
                per_class_localization={k:dict(matched=matched[k],total=n,recall=matched[k]/n)
                                        for k,n in sorted(truth.items())},
                false_positives_overlapping_context=dict(Counter(
                    context for _,label,context in matches if label is None and context
                )))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','archive','output'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--max-per-recording',type=int)
    args = parser.parse_args()
    metadata,frames,crops = load_dataset(args.dataset)
    issues = validate(metadata,frames,crops)
    if any(value for value in issues.values() if isinstance(value,list)):
        raise ValueError('Resolve split leakage flags before benchmarking.')
    if checksum(args.archive) != metadata['archive_sha256']:
        raise ValueError('Archive checksum differs from the manifest.')
    release = read_json(ROOT/'reference/instrument-video-v3-results.json')
    weight = ROOT/'models/video-instruments.pt'
    if checksum(weight) != release['checkpoint_sha256']:
        raise ValueError('Unexpected detector checkpoint.')
    model = YOLO(str(weight),task='detect')
    torch.set_num_threads(4)
    per_source = Counter()
    samples = []
    cutoff = release['comparison']['pilot']['score_threshold']
    with ZipFile(args.archive) as archive:
        for row in frames:
            recording = row['recording']
            if row['split']!='test' or (args.max_per_recording and per_source[recording]>=args.max_per_recording):
                continue
            _,image = image_from_archive(archive,row['member'])
            boxes = model.predict(image,imgsz=release['image_size'],conf=.001,max_det=300,
                                  device='cpu',verbose=False)[0].boxes
            positives = [a for a in row['annotations'] if a['label']!='non_instrument']
            negatives = [a for a in row['annotations'] if a['label']=='non_instrument']
            samples.append((positives,list(zip(boxes.conf.tolist(),boxes.xyxyn.tolist())),negatives))
            per_source[recording] += 1
    result = dict(dataset=metadata['version'],archive_sha256=metadata['archive_sha256'],
                  checkpoint_sha256=release['checkpoint_sha256'],score_cutoff=cutoff,
                  ranking_score_floor=.001, max_per_recording=args.max_per_recording,
                  frames_by_recording=dict(per_source), **detection_metrics(samples,cutoff))
    write_json(args.output,result)
    print(result)


if __name__=='__main__':
    main()
