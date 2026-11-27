"""Offline EPIC TSN inference and FPVSum category transfer baseline."""
from hashlib import file_digest
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models.video import R3D_18_Weights,r3d_18

from .contract import StudentKit
from .model import device_for
from .video import windows


LABEL_MAP={
    'bike':('riding a bike','riding mountain bike'),
    'bikepolo':('riding a bike','riding mountain bike','hitting baseball'),
    'boxing':('punching person (boxing)','punching bag'),
    'horse':('riding or walking with horse','grooming horse'),
    'jumping':('high jump','long jump','triple jump','jumping into pool','jumpstyle dancing'),
    'longboarding':('skateboarding','hoverboarding','snowboarding','roller skating'),
    'motor':('motorcycling',),
    'parkour':('parkour','vault','gymnastics tumbling'),
    'plane':('paragliding','parasailing','skydiving','flying kite'),
    'rockclimbing':('rock climbing','ice climbing','climbing a rope','climbing ladder','climbing tree'),
    'scuba':('scuba diving','snorkeling'),
    'skate':('skateboarding','hoverboarding','riding scooter','roller skating'),
    'ski':('skiing (not slalom or crosscountry)','skiing crosscountry','skiing slalom','ski jumping','snowboarding'),
    'surfing':('surfing water','windsurfing','kitesurfing','surfing crowd'),
}


def load_verified(path,expected):
    if not path.is_file():
        raise ValueError(f'Missing offline model asset: {path}')
    with path.open('rb') as stream:
        if file_digest(stream,'sha256').hexdigest()!=expected:
            raise ValueError(f'Checkpoint hash mismatch: {path.name}')
    return torch.load(path,map_location='cpu',weights_only=True)


def frame_tensor(image):
    h,w=image.shape[:2]
    factor=256/min(h,w)
    resized=cv2.resize(image,(round(w*factor),round(h*factor)),interpolation=cv2.INTER_LINEAR)
    y,x=(resized.shape[0]-224)//2,(resized.shape[1]-224)//2
    rgb=cv2.cvtColor(resized[y:y+224,x:x+224],cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2,0,1)


def load_tsn_calibration(path, base_sha256):
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(artifact, dict) or artifact.get('format') != 'egosight-tsn-calibration-v1'
            or artifact.get('pretrained_sha256') != base_sha256):
        raise ValueError('Calibration checkpoint does not match the source TSN weights')
    bias, log_scale = artifact.get('bias'), artifact.get('log_scale')
    if not all(isinstance(t, torch.Tensor) and t.shape == shape and torch.is_floating_point(t)
               and bool(torch.isfinite(t).all()) for t, shape in ((bias, (397,)), (log_scale, (2,)))):
        raise ValueError('Invalid TSN calibration parameters')
    scale = log_scale.exp().repeat_interleave(torch.tensor([97, 300]))
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Invalid TSN calibration scale')
    return scale, bias


class EpicTSN:
    def __init__(self,root,device,calibration=None):
        manifest=json.loads((root/'receipt.json').read_text())
        annotation=root/'epic-classes.json'
        with annotation.open('rb') as stream:
            if file_digest(stream,'sha256').hexdigest()!=manifest['epic-classes.json']:
                raise ValueError('EPIC class and pair metadata hash mismatch')
        pairs=json.loads(annotation.read_text())
        StudentKit().validate_classes(pairs['classes'])
        self.classes=pairs['classes']
        self.pairs=np.asarray(pairs['pairs'],dtype=int)
        if len(self.classes['verb'])!=97 or len(self.classes['noun'])!=300:
            raise ValueError('EPIC checkpoint requires 97 verbs and 300 nouns')
        self.device=device_for(device)
        state=load_verified(root/'tsn_rgb.ckpt',manifest['tsn_rgb.ckpt'])
        if not (state['hyper_parameters']['model']['type']=='TSN' and
                state['hyper_parameters']['model']['backbone']=='resnet50'):
            raise ValueError('Unexpected EPIC backbone')
        self.model=resnet50(weights=None)
        self.model.fc=nn.Linear(2048,397)
        prefix='model.base_model.'
        weights={key.removeprefix(prefix):value for key,value in state['state_dict'].items() if key.startswith(prefix)}
        weights.update({field:state['state_dict']['model.new_fc.'+field.split('.')[-1]]
                        for field in ('fc.weight','fc.bias')})
        self.model.load_state_dict(weights)
        self.model.to(self.device).eval()
        self.checksum=manifest['tsn_rgb.ckpt']

        self.calibration = None
        if calibration is not None:
            self.calibration = load_tsn_calibration(calibration, self.checksum)

    def features(self,reader):
        """Decode once at half-second cadence and cache only logits and motion."""
        capture=cv2.VideoCapture(str(reader.path))
        stride=max(1,round(reader.info.fps/2))
        detections,frames,times,motion=[],[],[],[]
        previous=None
        scale=torch.tensor([.485,.456,.406],device=self.device)[None,:,None,None]
        def flush():
            if frames:
                images=torch.stack(frames).to(self.device).float()/255
                detections.append(self.model((images-scale)/scale).cpu())
                frames.clear()
        try:
            with torch.inference_mode():
                for index in range(reader.info.frames):
                    if not capture.grab():
                        raise ValueError(f'Truncated video at frame {index}')
                    if index%stride:
                        continue
                    ok,image=capture.retrieve()
                    if not ok or image.shape[:2]!=(reader.info.height,reader.info.width):
                        raise ValueError(f'Invalid source frame {index}')
                    small=cv2.resize(image,(48,48))
                    motion.append(float(cv2.absdiff(small,previous).mean()/255) if previous is not None else 0.)
                    previous=small
                    frames.append(frame_tensor(image));times.append(index/reader.info.fps)
                    if len(frames)==8:
                        flush()
                flush()
        finally:
            capture.release()
        return np.asarray(times),torch.cat(detections),np.asarray(motion)

    def predict(self,reader):
        times,logits,motion=self.features(reader)
        actions=[]; candidates=[]; evidence=[]
        for length,stride in ((2.,1.),(4.,2.)):
            for start,end in windows(reader.info.duration,length,stride):
                indices=np.flatnonzero((times>=start)&(times<end))
                if not len(indices):
                    continue
                mean=logits[indices].mean(0)
                if self.calibration is not None:
                    scale, bias = self.calibration
                    mean = mean * scale + bias
                verb=mean[:97].softmax(0).numpy()
                noun=mean[97:].softmax(0).numpy()
                probabilities=verb[self.pairs[:,0]]*noun[self.pairs[:,1]]
                ranking=np.argsort(-probabilities,kind='stable')[:3]
                options=[]
                for i in ranking:
                    v,n=self.pairs[i]
                    options.append(dict(verb=self.classes['verb'][v],noun=self.classes['noun'][n],
                                        confidence=float(probabilities[i])))
                best=dict(start=start,end=end,visual=float(motion[indices].mean()),**options[0])
                actions.append(best)
                evidence.append(dict(start=start,end=end,alternatives=options))
                if length==4.:
                    candidates.append(best)
        from .proposals import process_proposals
        actions,action_metadata,proposal_stats=process_proposals([],reader.info.duration,'cooking',custom_actions=actions)
        return actions,candidates,'unclassified',0.,dict(model='EPIC-KITCHENS-100 TSN RGB ResNet50',
            checkpoint_sha256=self.checksum,cached_samples=len(times),window_alternatives=evidence,
            action_metadata=action_metadata,proposal_stats=proposal_stats)


class FPVTransfer:
    def __init__(self,root,device):
        manifest=json.loads((root/'receipt.json').read_text())
        self.checksum=manifest['r3d_18-b3b3357e.pth']
        self.device=device_for(device)
        self.model=r3d_18(weights=None)
        self.model.load_state_dict(load_verified(root/'r3d_18-b3b3357e.pth',self.checksum))
        self.model.to(self.device).eval()
        self.transform=R3D_18_Weights.DEFAULT.transforms()
        self.categories=R3D_18_Weights.DEFAULT.meta['categories']
        self.mapping={name:[self.categories.index(label) for label in labels] for name,labels in LABEL_MAP.items()}

    def predict(self,reader):
        candidates=[]; evidence=[]; aggregate=[]; window_scores=[]
        with torch.inference_mode():
            for start,end in windows(reader.info.duration,4.,2.):
                frames,_=reader.sample(start,end,16,128)
                motion=float(np.mean([np.abs(b.astype(float)-a.astype(float)).mean()/255 for a,b in zip(frames,frames[1:])]))
                inputs=self.transform(torch.from_numpy(np.stack(frames)).permute(0,3,1,2)).unsqueeze(0)
                probability=self.model(inputs.to(self.device)).softmax(1).cpu()[0]
                scores={name:float(probability[classes].sum()) for name,classes in self.mapping.items()}
                aggregate.append(scores)
                best=max(scores,key=scores.get)
                window_scores.append((start,end,scores,motion))
                candidates.append(dict(start=start,end=end,confidence=scores[best],visual=motion,verb='other',noun='other'))
                evidence.append(dict(start=start,end=end,category_scores=scores))

        # Multi-duration candidate generation (e.g. 6.0s intervals around high activity moments)
        if reader.info.duration >= 45.0:
            for i in range(len(window_scores) - 1):
                w1 = window_scores[i]
                w2 = window_scores[i + 1]
                s, e = w1[0], w2[1]
                if e <= reader.info.duration and (e - s) >= 5.9:
                    conf6 = (max(w1[2].values(), default=0.0) + max(w2[2].values(), default=0.0)) / 2.0
                    mot6 = (w1[3] + w2[3]) / 2.0
                    if conf6 >= 0.35 and mot6 >= 0.04:
                        candidates.append(dict(start=s, end=e, confidence=conf6, visual=mot6, verb='other', noun='other'))

        pooled={name:float(np.mean([row[name] for row in aggregate])) for name in self.mapping}
        label=max(pooled,key=pooled.get)
        score=pooled[label]
        if score<.04:
            label,score='other',0.
        from .proposals import process_proposals
        actions,action_metadata,proposal_stats=process_proposals(window_scores,reader.info.duration,label,min_conf=.10,iou_thresh=.30)
        return actions,candidates,label,score,dict(model='Torchvision R3D-18 Kinetics-400 transfer',
            checkpoint_sha256=self.checksum,category_scores=pooled,window_category_scores=evidence,
            unsupported_labels=sorted(StudentKit().labels-set(self.mapping)-{'cooking','other'}),
            action_metadata=action_metadata,proposal_stats=proposal_stats)
