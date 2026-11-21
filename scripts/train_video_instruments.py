"""Extract reviewed local clips and train a single-class video localization pilot."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import shutil
from time import perf_counter

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def capture_sources(root):
    return {folder.name: folder / 'source.mp4' for folder in root.iterdir()
            if folder.is_dir() and (folder/'source.mp4').is_file()}


def prepare(destination, source_root, second, stills):
    spec = json.loads((ROOT/'reference/xr4-instrument-frames.json').read_text())['sources']
    sources = capture_sources(source_root)
    sources['5892118'] = second
    for name, group in spec.items():
        subset = group['split']
        for fmt in ('images','labels'):
            (destination/subset/fmt).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(sources[name]))
        try:
            all_frames = group.get('frames') or group['negative_crops']
            for index, coordinates in all_frames.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(index))
                ok, im = cap.read()
                if not ok:
                    raise ValueError(f'Could not decode {name} frame {index}.')
                labels = []
                if 'negative_crops' in group:
                    h,w = im.shape[:2]
                    x,y,r,b = [round(v * d / span) for v,d,span in zip(coordinates,(w,h,w,h),(500,280,500,280))]
                    im=im[y:b,x:r]
                else:
                    for x,y,r,b in coordinates:
                        labels.append(f'0 {(x+r)/1000:.7f} {(y+b)/560:.7f} {(r-x)/500:.7f} {(b-y)/280:.7f}')
                copies = 3 if subset=='train' else 1
                for n in range(copies):
                    key=f'{name}-{index}-{n}'
                    cv2.imwrite(str(destination/subset/'images'/f'{key}.jpg'),im)
                    (destination/subset/'labels'/f'{key}.txt').write_text('\n'.join(labels)+'\n' if labels else '')
        finally:
            cap.release()
    photos = sorted((stills/'train/images').iterdir())
    for photo in photos[::max(1,len(photos)//100)]:
        src = stills/'train/labels'/f'{photo.stem}.txt'
        aabb=[]
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            coordinates=np.asarray(line.split()[1:],dtype=float).reshape(4,2)
            left,top=coordinates.min(axis=0)
            right,bottom=coordinates.max(axis=0)
            aabb.append(f'0 {(left+right)/2:.7f} {(top+bottom)/2:.7f} {right-left:.7f} {bottom-top:.7f}')
        key='still-'+photo.stem
        shutil.copyfile(photo,destination/'train/images'/f'{key}{photo.suffix}')
        (destination/'train/labels'/f'{key}.txt').write_text('\n'.join(aabb)+'\n' if aabb else '')
    data=destination/'data.yaml'
    data.write_text(yaml.safe_dump(dict(path=str(destination.resolve()),train='train/images',val='valid/images',
                                       test='test/images',names={0:'Dental instrument'})))
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sources',type=Path)
    p.add_argument('bedside_video',type=Path)
    p.add_argument('stills',type=Path)
    p.add_argument('backbone',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--epochs',type=int,default=12)
    opt=p.parse_args()
    if opt.output.exists():
        p.error('Use an empty output path.')
    torch.set_num_threads(6)
    data=prepare(opt.output/'data',opt.sources,opt.bedside_video,opt.stills)
    m=YOLO(str(opt.backbone.resolve()))
    t=perf_counter()
    results=m.train(data=str(data.resolve()),imgsz=512,epochs=opt.epochs,batch=12,workers=0,device='cpu',
                    project=str(opt.output.resolve()),name='fit',optimizer='AdamW',lr0=.001,freeze=10,amp=False,plots=False,
                    seed=44,mosaic=0,flipud=0,degrees=8)
    best=Path(results.save_dir)/'weights/best.pt'
    m=YOLO(str(best))
    test=m.val(data=str(data.resolve()),split='test',imgsz=512,device='cpu',workers=0,plots=False,
               project=str(opt.output.resolve()),name='test')
    report=dict(weights_sha256=file_digest(best.open('rb'),'sha256').hexdigest(),image_size=512,
                device='cpu',epochs=opt.epochs,test={key:float(value) for key,value in test.results_dict.items()},
                duration_seconds=round(perf_counter()-t,2),validation_sources=['6502766'],
                heldout_sources=['braces','33337556'],positive_class='Dental instrument')
    (opt.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
