"""Validation-only detector window geometry and cross-window merging."""
from itertools import product

import torch
from torchvision.ops import nms


def windows(width,height,method):
    if method=='full':
        return [(0,0,width,height)]
    if method=='roi':
        return [(0,0,width,height),(width//5,height//5,width-width//5,height-height//5)]
    if method!='tiles':
        raise ValueError('Unknown detector window method.')
    w,h=max(1,round(width*.65)),max(1,round(height*.65))
    return [(x,y,x+w,y+h) for x,y in product(sorted({0,width-w}),sorted({0,height-h}))]


def remap(box, window, width, height):
    x,y,right,bottom=window
    offsets=(x,y,x,y)
    extents=(right-x,bottom-y,right-x,bottom-y)
    return [(offset+max(0.,min(1.,value))*extent)/axis
            for value,offset,extent,axis in zip(box,offsets,extents,(width,height,width,height))]


def merge(predictions, overlap=.50):
    if not predictions:
        return []
    # Stable pre-order gives the same tie order across runs on CPU.
    scores=torch.tensor([score for score,_ in predictions],dtype=torch.float64)
    boxes=torch.tensor([box for _,box in predictions],dtype=torch.float64)
    order=nms(boxes,scores,overlap)
    return [predictions[index] for index in order.tolist()]


def predict_windows(model,image,method,size,score_floor=.001):
    height,width=image.shape[:2]
    regions=windows(width,height,method)
    crops=[image[y:b,x:r] for x,y,r,b in regions]
    results=model.predict(crops,imgsz=size,conf=score_floor,max_det=300,device='cpu',verbose=False)
    raw=[]
    for region,result in zip(regions,results):
        for score,box in zip(result.boxes.conf.tolist(),result.boxes.xyxyn.tolist()):
            source=remap(box,region,width,height)
            if source[0]<source[2] and source[1]<source[3]:
                raw.append((score,source))
    return (raw if method=='full' else merge(raw)),raw
