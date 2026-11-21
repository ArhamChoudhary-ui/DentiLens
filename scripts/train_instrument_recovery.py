"""Fit an experimental RGB head using individually approved training crops."""
import argparse
from collections import defaultdict, Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import cv2
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from instrument_dataset import ROOT, checksum, image_from_archive, load_dataset, read_json, validate, write_json
from instrument_identity import crop_tensor, encoded_crop, identity_model, model_pixels, network, score_crops
from evaluate_classifier import classification_metrics

CLASSES = ['dental_mirror','dental_explorer','non_instrument','unknown']
LIMITS = {'dental_mirror':4,'dental_explorer':4,'non_instrument':4,'unknown':2}


def shortlist(frames,crops):
    parents = {row['id']:row for row in frames}
    groups = defaultdict(list)
    for row in crops:
        parent = parents[row['parent']]
        label = row['label']
        box = row['box']
        if (row['split']=='train' and label in LIMITS
                and (label!='non_instrument' or row['original_class']=='hand_gloved')
                and max(box[2]-box[0],box[3]-box[1]) < .23):
            groups[parent['recording'],label].append(row)
    return [row for (_,label),rows in sorted(groups.items())
            for row in sorted(rows,key=lambda item:sha256(item['id'].encode()).digest())[:LIMITS[label]]]


def review_sheet(rows, image_for, output):
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'shortlist.json',[row['id'] for row in rows])
    for start in range(0,len(rows),60):
        page = rows[start:start+60]
        canvas = np.full((120*((len(page)+7)//8),116*8,3),32,np.uint8)
        for i,row in enumerate(page):
            y,x = (i//8)*120,(i%8)*116
            rgb = encoded_crop(image_for(row),row['box'],{'color':True})
            canvas[y:y+96,x:x+96] = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            label = {'dental_mirror':'mirror','dental_explorer':'explorer',
                     'non_instrument':'glove','unknown':'unknown'}[row['label']]
            cv2.putText(canvas,f'{start+i} {label}',(x+2,y+112),0,.42,(255,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(output/f'sheet-{start//60+1}.jpg'),canvas)


def reviewed_rows(crops, approvals):
    parents = {row['id']:row for row in crops}
    selected = []
    for identifier in approvals:
        row = parents.get(identifier)
        if not row or row['split']!='train' or row['label'] not in CLASSES:
            raise ValueError(f'Invalid training approval: {identifier}')
        selected.append(row)
    if not selected:
        raise ValueError('At least one approved training crop is required.')
    if len({row['id'] for row in selected})!=len(selected):
        raise ValueError('Repeated training approval.')
    return selected


def fit(rows, eval_rows, image_for, output, epochs, resume, provenance):
    torch.manual_seed(44)
    torch.set_num_threads(4)
    foundation, source = identity_model(ROOT/'models','instrument-detail')
    model = network(len(CLASSES),'mobilenet_v3_small')
    model.features.load_state_dict(foundation.features.state_dict())
    model.features.eval()
    history = []
    optimizer = torch.optim.AdamW(model.classifier.parameters(),lr=.001)
    best, start, stale = float('inf'),0,0
    previous_duration = 0.0
    if resume:
        state = torch.load(output/'last.pt',map_location='cpu',weights_only=True)
        model.classifier.load_state_dict(state['head'])
        optimizer.load_state_dict(state['optimizer'])
        best,start,stale,history = (state[key] for key in ('best','epoch','stale','history'))
        previous_duration = state.get('head_training_seconds',0.0)
    else:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/'provenance.json',provenance)
    def features(items):
        results, labels = [], []
        with torch.inference_mode():
            for low in range(0,len(items),32):
                batch = items[low:low+32]
                pixels = [encoded_crop(image_for(row),row['box'],{'color':True}) for row in batch]
                encoded = model.features(model_pixels(crop_tensor(pixels),'mobilenet_v3_small'))
                results.append(model.avgpool(encoded).flatten(1))
                labels.extend(CLASSES.index(row['label']) for row in batch)
        return torch.cat(results).clone(), torch.tensor(labels)
    preparation = perf_counter()
    train_x, train_y = features(rows)
    validation_x, validation_y = features(eval_rows)
    preparation_seconds = perf_counter()-preparation
    weights = torch.bincount(train_y,minlength=len(CLASSES)).float().clamp(min=1).pow(-.5)
    objective = nn.CrossEntropyLoss(weight=weights)
    clock = perf_counter()
    limit = start if stale>=4 else epochs
    for epoch in range(start,limit):
        torch.manual_seed(44+epoch)
        model.classifier.train()
        losses = []
        for values,y in DataLoader(TensorDataset(train_x,train_y),batch_size=24,shuffle=True):
            optimizer.zero_grad()
            loss = objective(model.classifier(values),y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.classifier.eval()
        with torch.inference_mode():
            loss = float(objective(model.classifier(validation_x),validation_y))
        history.append(dict(epoch=epoch+1,training_loss=float(np.mean(losses)),validation_loss=loss))
        print(history[-1],flush=True)
        if loss<best:
            best,stale = loss,0
            torch.save(model.state_dict(),output/'best.pt')
        else:
            stale+=1
        torch.save(dict(head=model.classifier.state_dict(),optimizer=optimizer.state_dict(),
                        best=best,epoch=epoch+1,stale=stale,history=history,
                        head_training_seconds=previous_duration+perf_counter()-clock),output/'last.pt')
        if stale>=4:
            break
    info = dict(model='instrument-recovery-mobilenet-v1-experimental', classes=CLASSES,
                backbone='mobilenet_v3_small',color=True,
                thresholds={'dental_mirror':.95,'dental_explorer':.95})
    model.load_state_dict(torch.load(output/'best.pt',map_location='cpu',weights_only=True))
    with torch.inference_mode():
        prediction = model.classifier(validation_x).argmax(1)
    write_json(output/'training.json',dict(model=info,foundation_sha256=source['weights_sha256'],
               checkpoint_sha256=checksum(output/'best.pt'),history=history,early_stopping_patience=4,
               duration_seconds=previous_duration+perf_counter()-clock, feature_seconds=preparation_seconds, validation_accuracy=float((prediction==validation_y).float().mean()),
               training_class_counts=dict(Counter(row['label'] for row in rows)),hardware='Apple Silicon CPU',
               seed=44,optimizer='AdamW',learning_rate=.001,batch=24))
    return model.eval(),info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('dataset','archive','output'):
        parser.add_argument(field,type=Path)
    parser.add_argument('--approvals',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--epochs',type=int,default=12)
    args=parser.parse_args()
    metadata,frames,crops=load_dataset(args.dataset)
    if any(v for v in validate(metadata,frames,crops).values() if isinstance(v,list)):
        raise ValueError('Resolve leakage before using this dataset.')
    if checksum(args.archive)!=metadata['archive_sha256']:
        raise ValueError('Archive mismatch.')
    parents={row['id']:row for row in frames}
    with ZipFile(args.archive) as archive:
        def image_for(row):
            return image_from_archive(archive,parents[row['parent']]['member'])[1]
        if args.approvals is None:
            review_sheet(shortlist(frames,crops),image_for,args.output)
            return
        approved=reviewed_rows(shortlist(frames,crops),read_json(args.approvals))
        provenance=dict(dataset=metadata['version'],archive=metadata['archive_sha256'],
                        manifests={name:checksum(args.dataset/name) for name in
                                   ('manifests/dataset.json','intraoral_detection/frames.json',
                                    'instrument_classification/crops.json')},
                        approvals_sha256=checksum(args.approvals),training_ids=[row['id'] for row in approved])
        if args.resume and read_json(args.output/'provenance.json')!=provenance:
            raise ValueError('Training input changed since the checkpoint was created.')
        valid=[row for row in crops if row['split']=='valid' and row['label'] in CLASSES]
        model,info=fit(approved,valid,image_for,args.output,args.epochs,args.resume,provenance)
        write_json(args.output/'provenance.json',provenance)
        test=[row for row in crops if row['split']=='test' and row['label'] in CLASSES]
        results=[]
        for low in range(0,len(test),32):
            batch=test[low:low+32]
            results.extend(score_crops(model,info,[encoded_crop(image_for(r),r['box'],info) for r in batch]))
        truth = [row['label'] for row in test]
        predicted = [prediction['class_id'] for prediction in results]
        report = classification_metrics(truth,results)
        report['native_four_way'] = dict(class_order=CLASSES, confusion_matrix=confusion_matrix(
            truth,predicted,labels=CLASSES).tolist(), classification_report=classification_report(
            truth,predicted,labels=CLASSES,output_dict=True,zero_division=0))
        write_json(args.output/'test.json',report)
        write_json(args.output/'errors.json',[
            dict(crop=row['id'],source=parents[row['parent']]['recording'],box=row['box'],
                 expected=row['label'],prediction=prediction)
            for row,prediction in zip(test,results)
            if prediction['class_id']!=row['label']
        ])


if __name__=='__main__':
    main()
