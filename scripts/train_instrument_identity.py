"""Train a small grayscale crop classifier from V3 and reviewed local frames."""
import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
import random
from pathlib import Path
import sys
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from instrument_identity import crop, crop_tensor, encode, model_pixels, network

ALIASES = {
    'Dental Mirrors':'dental_mirror','Mouth mirror':'dental_mirror','Explorer No.5':'dental_explorer',
    'Artery Forceps':'dental_forceps','Lower Molar Forceps':'dental_forceps',
    'airotor':'dental_handpiece','Contra-angle handpiece':'dental_handpiece',
    'Weingart Pliers':'orthodontic_pliers','Ligature Applicator':'ligature_applicator',
    'Hard Wire Cutter':'wire_cutter','Distal End Cutter':'wire_cutter','Bracket Holder':'bracket_holder',
}
VIDEO_ALIASES = {
    **ALIASES, 'Artery Forceps':'hemostat', 'Distal End Cutter':'distal_end_cutter',
    'Band Pusher':'band_pusher', 'Bayonet Foreceps':'bayonet_forceps', 'Blade holder':'blade_holder',
    'Bracket Positioner':'bracket_positioner', 'Burnisher':'burnisher', 'Cord packer':'cord_packer',
    'Dycal carrier':'dycal_carrier', 'Rubber Dam Forceps':'rubber_dam_forceps',
    'Suture Scissors':'suture_scissors', 'Toothed Tissue Forceps':'tissue_forceps',
    'spoon excavators':'spoon_excavator', 'triple syringe dental':'three_way_syringe',
}


def photo_data(path, aliases, color):
    results = {split:[] for split in ('train','valid','test')}
    seen = set()
    with ZipFile(path) as archive:
        classes = yaml.safe_load(archive.read('data.yaml'))['names']
        names = set(archive.namelist())
        for split in ('test','valid','train'):
            groups = set()
            for path in sorted(names):
                if not path.startswith(split+'/labels/') or not path.endswith('.txt'):
                    continue
                stem = Path(path).stem
                group = stem.split('.rf.',1)[0]
                if group in groups or group in seen:
                    continue
                image_path = f'{split}/images/{stem}.jpg'
                if image_path not in names:
                    continue
                payload = archive.read(image_path)
                digest = sha256(payload).hexdigest()
                if digest in seen:
                    continue
                image = cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR)
                if image is None:
                    continue
                groups.add(group)
                seen.add(digest)
                for line in archive.read(path).decode().splitlines():
                    fields = line.split()
                    if len(fields)!=9:
                        continue
                    code = int(fields[0])
                    corners = np.asarray(fields[1:],dtype=float).reshape(4,2).clip(0,1)
                    region = crop(image,[*corners.min(axis=0),*corners.max(axis=0)])
                    if region.size and min(region.shape[:2])>=5:
                        results[split].append((encode(region,color), aliases.get(classes[code],'unknown')))
            seen.update(groups)
    return results


def video_data(source, selections, grid, color, augment=False):
    cap = cv2.VideoCapture(str(source))
    rows = []
    try:
        for index, objects in selections.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(index))
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Could not decode {source.name} frame {index}.')
            for bounds,label in objects:
                region = crop(image,np.asarray(bounds)/np.tile(grid,2))
                rows.append((encode(region,color),label))
                if augment:
                    for padding in (0,.24):
                        variant = crop(image,np.asarray(bounds)/np.tile(grid,2),padding=padding)
                        rows.append((encode(variant,color),label))
                    rows.append((encode(cv2.flip(region,1),color),label))
                    faded = cv2.convertScaleAbs(region,alpha=.78,beta=24)
                    rows.append((encode(faded,color),label))
    finally:
        cap.release()
    return rows


def tensors(rows, classes):
    return TensorDataset(crop_tensor([image for image,_ in rows]),
                         torch.tensor([classes.index(label) for _,label in rows]))


def measure(model, data, backbone):
    truth,pred,confidence = [],[],[]
    with torch.inference_mode():
        for data,labels in DataLoader(data,batch_size=64):
            scores,indices = model(model_pixels(data,backbone) if backbone else data).softmax(1).max(dim=1)
            truth.extend(labels.tolist()); pred.extend(indices.tolist()); confidence.extend(scores.tolist())
    return np.asarray(truth),np.asarray(pred),np.asarray(confidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive','manifest','output'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--epochs',type=int,default=14)
    parser.add_argument('--foundation',type=Path)
    parser.add_argument('--adapt-video',action='store_true')
    parser.add_argument('--mirror-explorer',action='store_true')
    opt=parser.parse_args()
    torch.set_num_threads(6)
    torch.manual_seed(44)
    opt.output.mkdir(parents=True,exist_ok=False)
    if opt.adapt_video and not opt.foundation:
        parser.error('--adapt-video requires pretrained MobileNet weights.')
    if opt.mirror_explorer and not opt.adapt_video:
        parser.error('--mirror-explorer requires --adapt-video.')
    aliases=VIDEO_ALIASES if opt.adapt_video else ALIASES
    target={'dental_mirror','dental_explorer'}
    if opt.mirror_explorer:
        aliases={name:label for name,label in aliases.items() if label in target}
    data=photo_data(opt.archive, aliases, opt.adapt_video)
    if opt.mirror_explorer:
        unknown=[row for row in data['train'] if row[1]=='unknown']
        data['train']=[row for row in data['train'] if row[1]!='unknown']+random.Random(44).sample(unknown,min(650,len(unknown)))
    photo_count=len(data['train'])
    videos=json.loads(opt.manifest.read_text())
    for source in videos:
        examples=video_data((opt.manifest.parent/source['file']).resolve(),source['frames'],source['grid'],
                            opt.adapt_video,augment=opt.adapt_video and source['split']=='train')
        if opt.mirror_explorer:
            examples=[(pixels,label if label in target else 'unknown') for pixels,label in examples]
        data.setdefault(source['split'],[]).extend(examples)
    classes=sorted({label for rows in data.values() for _,label in rows} | {'unknown'})
    print({k:Counter(label for _,label in rows) for k,rows in data.items()},flush=True)
    backbone='mobilenet_v3_small' if opt.foundation else 'cnn'
    model=network(len(classes),backbone)
    tensors_by_split={split:tensors(rows,classes) for split,rows in data.items()}
    if opt.foundation:
        foundation=torch.load(opt.foundation,map_location='cpu',weights_only=True)
        model.load_state_dict({key:value for key,value in foundation.items()
                              if not key.startswith('classifier.3.')},strict=False)
        model.eval()
        with torch.inference_mode():
            for split,dataset in tensors_by_split.items():
                features=[]
                for images,_ in DataLoader(dataset,batch_size=64):
                    hidden=model.features(model_pixels(images,backbone))
                    features.append(model.avgpool(hidden).flatten(1))
                tensors_by_split[split]=TensorDataset(torch.cat(features),dataset.tensors[1])
        print('Extracted pretrained features on CPU',flush=True)
    train=tensors_by_split['train']
    counts=torch.bincount(train.tensors[1])
    weights=counts[train.tensors[1]].float().pow(-.65)
    if opt.adapt_video:
        weights[photo_count:] *= 5
    loader=DataLoader(train,batch_size=64,sampler=WeightedRandomSampler(weights,len(weights)))
    head=model.classifier if opt.foundation else model
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001)
    input_mode=None if opt.foundation else backbone
    best=-1
    path=opt.output/'instrument-identity.pt'
    for epoch in range(opt.epochs):
        head.train()
        for images, labels in loader:
            pixels=images if opt.foundation else model_pixels(images,backbone)
            if not opt.foundation:
                pixels=(pixels*torch.empty((len(images),1,1,1)).uniform_(.75,1.25)
                        +torch.empty((len(images),1,1,1)).uniform_(-.08,.08)).clamp(0,1)
                if torch.rand(1).item()<.5:
                    pixels=pixels.flip(-1)
            optimizer.zero_grad()
            loss=torch.nn.functional.cross_entropy(head(pixels),labels)
            loss.backward()
            optimizer.step()
        truth,pred,_=measure(head.eval(),tensors_by_split['valid'],input_mode)
        summary=classification_report(truth,pred,labels=list(range(len(classes))),output_dict=True,zero_division=0)
        supported=[item['f1-score'] for key,item in summary.items() if key.isdigit() and item['support']]
        f1=sum(supported)/len(supported)
        print(f'Epoch {epoch+1}: validation macro-F1 {f1:.3f}',flush=True)
        if f1>best:
            torch.save(model.state_dict(),path)
            best=f1
    model.load_state_dict(torch.load(path,weights_only=True))
    truth,pred,confidence=measure(head.eval(),tensors_by_split['valid'],input_mode)
    thresholds={}
    for i,label in enumerate(classes):
        if label=='unknown':
            continue
        for cutoff in (.75,.80,.85,.90,.95,.98):
            mask=(pred==i)&(confidence>=cutoff)
            if mask.sum()>=5 and (truth[mask]==i).mean()>=.90:
                thresholds[label]=cutoff
                break
    scores={}
    for split in ('valid','test','video_test'):
        if not data.get(split):
            continue
        truth,pred,confidence=measure(head.eval(),tensors_by_split[split],input_mode)
        verified=np.asarray([i if score>=thresholds.get(classes[i],2) else classes.index('unknown')
                             for i,score in zip(pred,confidence)])
        scores[split]=dict(crops=len(truth), raw_accuracy=float((pred==truth).mean()),
            abstention=float((verified==classes.index('unknown')).mean()),
            report=classification_report(truth,verified,labels=list(range(len(classes))),
                                         target_names=classes,output_dict=True,zero_division=0),
            confusion_matrix=confusion_matrix(truth,verified,labels=list(range(len(classes)))).tolist())
    with path.open('rb') as stream:
        checksum=file_digest(stream,'sha256').hexdigest()
    name='instrument-crop-mobilenet-v3-mirror-explorer' if opt.mirror_explorer else (
         'instrument-crop-mobilenet-v2' if opt.adapt_video else
         'instrument-crop-mobilenet-v1' if opt.foundation else 'instrument-crop-cnn-v1')
    info=dict(model=name,backbone=backbone,color=opt.adapt_video,classes=classes,thresholds=thresholds,aliases=aliases,
              weights_sha256=checksum,epochs=opt.epochs,scores=scores)
    path.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
    print('Thresholds',thresholds,flush=True)

if __name__=='__main__':
    main()
