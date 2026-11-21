"""Local classification of padded instrument proposals."""
from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import file_digest
import json

import cv2
import numpy as np
import torch
from torch import nn

LABELS = {
    'dental_mirror': 'Dental mirror', 'dental_explorer': 'Dental explorer',
    'periodontal_probe': 'Periodontal probe', 'dental_forceps': 'Dental forceps',
    'dental_scaler': 'Dental scaler', 'suction_tip': 'Suction tip',
    'dental_handpiece': 'Dental handpiece', 'orthodontic_pliers': 'Orthodontic pliers',
    'ligature_applicator': 'Ligature applicator', 'wire_cutter': 'Wire cutter',
    'bracket_holder': 'Bracket holder', 'other_instrument': 'Other instrument',
    'unknown': 'Uncertain instrument',
}
SIZE = 96


def crop(image, box, padding=.10):
    height, width = image.shape[:2]
    left, top, right, bottom = box
    dx, dy = (right-left)*padding, (bottom-top)*padding
    x0, y0 = max(0, int((left-dx)*width)), max(0, int((top-dy)*height))
    x1, y1 = min(width, int(np.ceil((right+dx)*width))), min(height, int(np.ceil((bottom+dy)*height)))
    return image[y0:y1, x0:x1]


def encode(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = SIZE/max(h, w)
    scaled = cv2.resize(gray, (max(1,round(w*scale)),max(1,round(h*scale))), interpolation=cv2.INTER_AREA)
    background = np.full((SIZE,SIZE), 127, dtype=np.uint8)
    y, x = (SIZE-scaled.shape[0])//2, (SIZE-scaled.shape[1])//2
    background[y:y+scaled.shape[0], x:x+scaled.shape[1]] = scaled
    return background


def network(classes, backbone='cnn'):
    if backbone=='mobilenet_v3_small':
        from torchvision.models import mobilenet_v3_small
        return mobilenet_v3_small(weights=None,num_classes=classes)

    layers = []
    last = 1
    for channels in (24, 48, 96):
        layers.extend([nn.Conv2d(last, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels),
                       nn.ReLU(), nn.MaxPool2d(2)])
        last = channels
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d((3,3)), nn.Flatten(), nn.Dropout(.20),
                         nn.Linear(96*9, classes))


@lru_cache(maxsize=2)
def load_identity(root, changed):
    del changed
    weights = root / 'instrument-identity.pt'
    info = json.loads(weights.with_suffix('.json').read_text())
    with weights.open('rb') as stream:
        if file_digest(stream,'sha256').hexdigest() != info['weights_sha256']:
            raise ValueError('Instrument classifier checksum mismatch.')
    classes = info['classes']
    if len(classes) != len(set(classes)) or 'unknown' not in classes or any(key not in LABELS for key in classes):
        raise ValueError('Instrument classifier class mapping is incompatible.')
    thresholds = info['thresholds']
    if any(key not in classes or key=='unknown' or not 0 < value <= 1 for key,value in thresholds.items()):
        raise ValueError('Instrument classifier thresholds are incompatible.')
    model = network(len(classes), info.get('backbone','cnn'))
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    return model.eval(), info


def identity_model(root):
    weights = root / 'instrument-identity.pt'
    metadata = weights.with_suffix('.json')
    if not weights.is_file() or not metadata.is_file():
        raise ValueError('Instrument classifier weights are missing.')
    return load_identity(root,(weights.stat().st_mtime_ns,metadata.stat().st_mtime_ns))


def model_pixels(data, backbone):
    pixels = data.float()/255
    if backbone=='mobilenet_v3_small':
        pixels = torch.nn.functional.interpolate(pixels,size=(128,128),mode='bilinear',align_corners=False)
        mean = torch.tensor([.485,.456,.406])[None,:,None,None]
        std = torch.tensor([.229,.224,.225])[None,:,None,None]
        pixels = (pixels-mean)/std
    return pixels


def classify(model, info, image, boxes):
    crops = [crop(image,box) for box in boxes]
    valid = [(i,encode(region)) for i,region in enumerate(crops) if region.size and min(region.shape[:2]) >= 5]
    output = [None] * len(boxes)
    if not valid:
        return output
    data = model_pixels(torch.from_numpy(np.stack([region for _,region in valid]))[:,None],info.get('backbone','cnn'))
    with torch.inference_mode():
        probabilities = model(data).softmax(1).numpy()
    for (index,_), scores in zip(valid,probabilities):
        order = np.argsort(-scores)[:3]
        top = [{'class_id':info['classes'][i], 'label':LABELS[info['classes'][i]], 'confidence':float(scores[i])}
               for i in order]
        result = top[0]
        accepted = result['class_id'] in info['thresholds'] and result['confidence'] >= info['thresholds'][result['class_id']]
        output[index] = {**result,'accepted':accepted,'model':info['model'],'alternatives':top}
    return output


def classify_observations(source, observations, model, info):
    by_frame = defaultdict(list)
    for item in observations:
        by_frame[item['frame']].append(item)
    output = {}
    capture = cv2.VideoCapture(str(source))
    try:
        position = 0
        for index, rows in sorted(by_frame.items()):
            for skipped in range(position,index):
                if not capture.grab():
                    raise ValueError(f'Could not decode frame {skipped}.')
            ok, image = capture.read()
            position = index + 1
            if not ok:
                raise ValueError(f'Could not classify frame {index}.')
            for row, identity in zip(rows,classify(model,info,image,[row['box'] for row in rows])):
                output[row['detection_id']] = identity
    finally:
        capture.release()
    # Keep predictions frame-by-frame for audit. A visible name also needs
    # repeated evidence on the same spatial track.
    for track in associate([dict(row) for row in observations]):
        identities = [output[row['detection_id']] for row in track]
        accepted = [row for row in identities if row and row['accepted']]
        votes = Counter(row['class_id'] for row in accepted)
        for identity in accepted:
            if votes[identity['class_id']] < 2 or votes[identity['class_id']] <= len(accepted)/2:
                identity['accepted'] = False
                identity['reason'] = 'insufficient_track_agreement'
    return output


def associate(rows):
    previous = []
    last_time = None
    track = 0
    grouped = {}
    by_time = {}
    for row in rows:
        by_time.setdefault(row['time'],[]).append(row)

    def matches(a,b):
        x,y,r,d = a['box']
        u,v,s,t = b['box']
        intersection = max(0,min(r,s)-max(x,u)) * max(0,min(d,t)-max(y,v))
        return intersection / max(1e-9,(r-x)*(d-y)+(s-u)*(t-v)-intersection) >= .45

    for time, current in sorted(by_time.items()):
        if last_time is None or time-last_time>.24:
            previous = []
        for row in current:
            options = [item for item in previous if matches(row,item)]
            if len(options)==1 and sum(matches(item,options[0]) for item in current)==1:
                row['track_id'] = options[0]['track_id']
            else:
                track += 1
                row['track_id'] = f'V-{track}'
            grouped.setdefault(row['track_id'],[]).append(row)
        previous,last_time = current,time
    return grouped.values()
