"""Build small-tool evidence, crop scale comparisons and a local contact sheet."""
import argparse
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import torch

from compare_detector_resolutions import small
from evaluate_video_instruments import iou
from instrument_dataset import ROOT, checksum, image_from_archive, load_dataset, read_json, write_json
from instrument_identity import classify, crop_bounds, identity_model, identity_pipeline


def card(canvas,image,box,index,label):
    h,w=image.shape[:2]
    cx,cy=(box[0]+box[2])*w/2,(box[1]+box[3])*h/2
    half=max(34,(box[2]-box[0])*w,(box[3]-box[1])*h)
    x,y,r,b=max(0,round(cx-half)),max(0,round(cy-half)),min(w,round(cx+half)),min(h,round(cy+half))
    scene=image[y:b,x:r].copy()
    inner=crop_bounds(box,w,h,0)
    cv2.rectangle(scene,(inner[0]-x,inner[1]-y),(inner[2]-x,inner[3]-y),(30,255,100),1)
    scale=min(232/scene.shape[1],148/scene.shape[0])
    thumb=cv2.resize(scene,(max(1,round(scene.shape[1]*scale)),max(1,round(scene.shape[0]*scale))),
                     interpolation=cv2.INTER_NEAREST if scale>1 else cv2.INTER_AREA)
    left,top=(index%3)*252,(index//3)*181
    canvas[top:top+thumb.shape[0],left:left+thumb.shape[1]]=thumb
    cv2.putText(canvas,f'{index+1:02} {label}',(left+3,top+169),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1)


def evidence(frame, annotation, ledgers, image, models):
    box=annotation['box']
    w,h=frame['width'],frame['height']
    width,height=(box[2]-box[0])*w,(box[3]-box[1])*h
    x,y,r,b=crop_bounds(box,w,h,0)
    patch=cv2.cvtColor(image[y:b,x:r],cv2.COLOR_BGR2GRAY)
    predictions={}
    for name,ledger in ledgers.items():
        row=ledger[frame['id']]
        nearest=max(((iou(box,region),score,region) for score,region in row['merged']),default=None)
        unmerged=max(((iou(box,region),score,region) for score,region in row['pre_merge']),default=None)
        predictions[name]=dict(nearest=nearest,best_frame_score=max((score for score,_ in row['merged']),default=None),
                               nearest_before_cross_window_merge=unmerged,
                               nearest_premerge_retained=any(unmerged and (score,region)==(unmerged[1],unmerged[2])
                                                             for score,region in row['merged']))
    crops={}
    for model,info in models:
        crops[info['model']]={}
        for padding in (0.,.1,.25,.35):
            result=classify(model,{**info,'crop_padding':padding},image,[box])[0]
            crops[info['model']][str(padding)]=result
    return dict(recording=frame['recording'],frame=frame['frame'],id=frame['id'],member=frame['member'],box=box,
                width=width,height=height,short_edge=min(width,height),long_edge=max(width,height),area=width*height,
                aspect_ratio=max(width,height)/min(width,height),label=annotation['label'],
                original_class=annotation['original_class'],visibility=annotation['visibility'],
                occlusion=annotation['occlusion'],native_laplacian_variance=float(cv2.Laplacian(patch,cv2.CV_64F).var()),
                detectors=predictions,classifiers=crops)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('dataset','archive','output'):
        parser.add_argument(field,type=Path)
    parser.add_argument('--runs',type=Path)
    parser.add_argument('--shadow',type=Path)
    args=parser.parse_args()
    metadata,frames,_=load_dataset(args.dataset)
    if checksum(args.archive)!=metadata['archive_sha256']:
        raise ValueError('Archive does not match the locked manifest.')
    torch.set_num_threads(4)
    models=identity_pipeline(ROOT/'models')
    if args.shadow:
        models.append(identity_model(args.shadow,'model'))
    ledgers={p.stem.removesuffix('-predictions'):read_json(p) for p in sorted(args.runs.glob('*-predictions.json'))} if args.runs else {}
    cases=[(frame,annotation) for frame in frames if frame['split']=='valid'
           for annotation in frame['annotations'] if annotation['label']!='non_instrument' and small(annotation['box'],frame)]
    canvas=np.full((181*((len(cases)+2)//3),756,3),28,np.uint8)
    report=[]
    args.output.mkdir(parents=True,exist_ok=False)
    with ZipFile(args.archive) as archive:
        for index,(frame,annotation) in enumerate(cases):
            _,image=image_from_archive(archive,frame['member'])
            card(canvas,image,annotation['box'],index,annotation['label'])
            report.append(evidence(frame,annotation,ledgers,image,models))
    write_json(args.output/'small-tools.json',report)
    cv2.imwrite(str(args.output/'contact-sheet.jpg'),canvas)
    print(Counter(row['label'] for row in report))


if __name__=='__main__':
    main()
