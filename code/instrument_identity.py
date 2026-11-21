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
    'band_pusher': 'Band pusher', 'bayonet_forceps': 'Bayonet forceps',
    'hemostat': 'Artery forceps', 'blade_holder': 'Blade holder',
    'bracket_positioner': 'Bracket positioner', 'burnisher': 'Burnisher',
    'cord_packer': 'Cord packer', 'distal_end_cutter': 'Distal end cutter',
    'dycal_carrier': 'Dycal carrier', 'rubber_dam_forceps': 'Rubber dam forceps',
    'suture_scissors': 'Suture scissors', 'tissue_forceps': 'Toothed tissue forceps',
    'spoon_excavator': 'Spoon excavator', 'three_way_syringe': 'Three-way syringe',
    'unknown': 'Uncertain instrument',
}
SIZE = 96
ASSOCIATION_VERSION = 2


def crop(image, box, padding=.10):
    height, width = image.shape[:2]
    left, top, right, bottom = box
    dx, dy = (right-left)*padding, (bottom-top)*padding
    x0, y0 = max(0, int((left-dx)*width)), max(0, int((top-dy)*height))
    x1, y1 = min(width, int(np.ceil((right+dx)*width))), min(height, int(np.ceil((bottom+dy)*height)))
    return image[y0:y1, x0:x1]


def encode(image, color=False):
    pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB if color else cv2.COLOR_BGR2GRAY)
    h, w = pixels.shape[:2]
    scale = SIZE/max(h, w)
    scaled = cv2.resize(pixels, (max(1,round(w*scale)),max(1,round(h*scale))), interpolation=cv2.INTER_AREA)
    background = np.full((SIZE,SIZE,3) if color else (SIZE,SIZE),127,dtype=np.uint8)
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
def load_identity(root, stem, changed):
    del changed
    weights = root / (stem+'.pt')
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


def identity_model(root, stem='instrument-identity'):
    weights = root / (stem+'.pt')
    metadata = weights.with_suffix('.json')
    if not weights.is_file() or not metadata.is_file():
        raise ValueError('Instrument classifier weights are missing.')
    return load_identity(root,stem,(weights.stat().st_mtime_ns,metadata.stat().st_mtime_ns))


def identity_pipeline(root):
    primary = identity_model(root)
    specialist_weights = root/'instrument-detail.pt'
    if specialist_weights.exists() or specialist_weights.with_suffix('.json').exists():
        return [primary,identity_model(root,'instrument-detail')]
    return [primary]


def pipeline_version(models):
    return [{'model':info['model'],'checksum':info['weights_sha256'],'thresholds':info['thresholds']}
            for _,info in models]


def classify_pipeline(source, observations, models):
    model, info = models[0]
    predictions = classify_observations(source,observations,model,info)
    if len(models)==1:
        return predictions
    specialist, detail = models[1]
    specialized = classify_observations(source,observations,specialist,detail)
    for key, prediction in specialized.items():
        if not prediction or not prediction['accepted']:
            continue
        earlier = predictions[key]
        if earlier and earlier['accepted'] and earlier['class_id']!=prediction['class_id']:
            predictions[key] = {**earlier,'accepted':False,'reason':'classifier_disagreement',
                                'specialist_prediction':prediction}
        else:
            predictions[key] = {**prediction,'primary_prediction':earlier}
    return predictions


def crop_tensor(images):
    stack = torch.from_numpy(np.stack(images))
    return stack.permute(0,3,1,2) if stack.ndim==4 else stack[:,None]


def model_pixels(data, backbone):
    pixels = data.float()/255
    if backbone=='mobilenet_v3_small':
        pixels = torch.nn.functional.interpolate(pixels,size=(128,128),mode='bilinear',align_corners=False)
        mean = torch.tensor([.485,.456,.406])[None,:,None,None]
        std = torch.tensor([.229,.224,.225])[None,:,None,None]
        pixels = (pixels-mean)/std
    return pixels


def encoded_crop(image, box, info):
    region = crop(image,box)
    if region.size and min(region.shape[:2]) >= 5:
        return encode(region,info.get('color',False))
    return None


def score_crops(model, info, crops):
    valid = [(index,pixels) for index,pixels in enumerate(crops) if pixels is not None]
    output = [None] * len(crops)
    if not valid:
        return output
    data = model_pixels(crop_tensor([pixels for _,pixels in valid]),info.get('backbone','cnn'))
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


def classify(model, info, image, boxes):
    return score_crops(model,info,[encoded_crop(image,box,info) for box in boxes])


def classify_observations(source, observations, model, info):
    by_frame = defaultdict(list)
    for item in observations:
        by_frame[item['frame']].append(item)
    output = {}
    pending = []

    def flush():
        result = score_crops(model,info,[pixels for _,pixels in pending])
        output.update((key,identity) for (key,_),identity in zip(pending,result))
        pending.clear()

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
            for row in rows:
                pending.append((row['detection_id'],encoded_crop(image,row['box'],info)))
                if len(pending)==32:
                    flush()
        if pending:
            flush()
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
        width, height, prev_width, prev_height = r-x, d-y, s-u, t-v
        if min(width,height,prev_width,prev_height) <= 0:
            return False
        intersection = max(0,min(r,s)-max(x,u)) * max(0,min(d,t)-max(y,v))
        overlap = intersection / (width*height+prev_width*prev_height-intersection)
        if overlap >= .45:
            return True
        # Keep a moving box associated when size and shape agree. Still
        # require overlap and a unique match in both directions.
        if overlap < .17 or not (.64 <= width/prev_width <= 1.56 and .64 <= height/prev_height <= 1.56):
            return False
        dx = (x+r-u-s)/(width+prev_width)
        dy = (y+d-v-t)/(height+prev_height)
        return dx*dx + dy*dy <= .66**2

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
