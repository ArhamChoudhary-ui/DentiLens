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
    'hand_gloved': 'Gloved hand', 'hand_ungloved': 'Bare hand', 'mouth_open': 'Open mouth',
    'unknown': 'Uncertain instrument',
}
SIZE = 96
ASSOCIATION_VERSION = 6
MAX_TRACK_GAP = .43
CONTEXT_CLASSES = {'hand_gloved','hand_ungloved','mouth_open'}


DEFAULT_CROP_PADDING = .10


def crop_bounds(box, width, height, padding=DEFAULT_CROP_PADDING):
    left, top, right, bottom = box
    dx, dy = (right-left)*padding, (bottom-top)*padding
    return (max(0,int((left-dx)*width)), max(0,int((top-dy)*height)),
            min(width,int(np.ceil((right+dx)*width))), min(height,int(np.ceil((bottom+dy)*height))))


def crop_geometry(box,width,height,padding=DEFAULT_CROP_PADDING):
    x0,y0,x1,y1 = crop_bounds(box,width,height,padding)
    w,h = max(0,x1-x0),max(0,y1-y0)
    scale = SIZE/max(1,w,h)
    return {'width':w,'height':h,'encoded_width':max(1,round(w*scale)) if w else 0,
            'encoded_height':max(1,round(h*scale)) if h else 0,'padding':padding}


def crop(image, box, padding=DEFAULT_CROP_PADDING):
    height,width = image.shape[:2]
    x0,y0,x1,y1 = crop_bounds(box,width,height,padding)
    return image[y0:y1,x0:x1]


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


@lru_cache(maxsize=3)
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
    padding = info.get('crop_padding',DEFAULT_CROP_PADDING)
    if not isinstance(padding,(int,float)) or not 0<=padding<=.35:
        raise ValueError('Instrument crop padding must be between 0 and 0.35.')
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
    return [{'model':info['model'],'checksum':info['weights_sha256'],'thresholds':info['thresholds'],
             'crop_padding':info.get('crop_padding',DEFAULT_CROP_PADDING)}
            for _,info in models]


def context_model(root):
    weights = root/'instrument-context.pt'
    if not weights.exists() and not weights.with_suffix('.json').exists():
        return None
    model,info = identity_model(root,'instrument-context')
    if set(info['classes']) - CONTEXT_CLASSES - {'unknown'}:
        raise ValueError('Non-instrument classifier class mapping is incompatible.')
    return model,info


def classify_pipeline(source, observations, models, context=None):
    raw = predict_video(source,observations,[*models,*([context] if context else [])])
    predictions = raw[0]
    for index, _ in enumerate(models[1:],start=1):
        for key, detail in raw[index].items():
            prior = predictions[key]
            if not detail:
                continue
            if not prior:
                predictions[key] = detail
            elif not detail['frame_accepted']:
                predictions[key] = {**prior,'specialist_prediction':detail}
            elif prior['frame_accepted'] and prior['class_id']!=detail['class_id']:
                predictions[key] = {**prior,'accepted':False,'frame_accepted':False,
                                    'reason':'classifier_disagreement','specialist_prediction':detail}
            else:
                predictions[key] = {**detail,'primary_prediction':prior}
    apply_consensus(observations,predictions)
    if context:
        evidence = raw[-1]
        apply_consensus(observations,evidence)
        for key,prediction in predictions.items():
            if prediction is not None:
                prediction['context'] = evidence[key]
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
    region = crop(image,box,info.get('crop_padding',DEFAULT_CROP_PADDING))
    if region.size and min(region.shape[:2]) >= 5:
        return encode(region,info.get('color',False))
    return None


def score_crops(model, info, crops, *, diagnostics=False):
    valid = [(index,pixels) for index,pixels in enumerate(crops) if pixels is not None]
    output = [None] * len(crops)
    if not valid:
        return output
    data = model_pixels(crop_tensor([pixels for _,pixels in valid]),info.get('backbone','cnn'))
    with torch.inference_mode():
        logits = model(data)
        probabilities = logits.softmax(1).numpy()
    for (index,_), scores, raw in zip(valid,probabilities,logits.tolist() if diagnostics else [None]*len(valid)):
        order = np.argsort(-scores)[:3]
        top = [{'class_id':info['classes'][i], 'label':LABELS[info['classes'][i]], 'confidence':float(scores[i])}
               for i in order]
        result = top[0]
        accepted = result['class_id'] in info['thresholds'] and result['confidence'] >= info['thresholds'][result['class_id']]
        reason = None if accepted else ('class_unknown' if result['class_id']=='unknown' else
                 'class_not_enabled' if result['class_id'] not in info['thresholds'] else 'below_class_threshold')
        output[index] = {**result,'accepted':accepted,'frame_accepted':accepted,'model':info['model'],
                         'reason':reason,'alternatives':top,
                         'probabilities':dict(zip(info['classes'],map(float,scores)))}
        if diagnostics:
            output[index].update(logits=raw, threshold=info['thresholds'].get(result['class_id']),
                                 margin=top[0]['confidence']-top[1]['confidence'] if len(top)>1 else None)
    return output


def classify(model, info, image, boxes):
    return score_crops(model,info,[encoded_crop(image,box,info) for box in boxes])


def predict_video(source, observations, models):
    by_frame = defaultdict(list)
    for item in observations:
        by_frame[item['frame']].append(item)
    output = [{} for _ in models]
    pending = []

    def flush():
        keys = [key for key,_ in pending]
        for index,(model,info) in enumerate(models):
            key = (info.get('color',False),info.get('crop_padding',DEFAULT_CROP_PADDING))
            result = score_crops(model,info,[pixels[key] for _,pixels in pending])
            output[index].update(zip(keys,result))
        pending.clear()

    variants = {(info.get('color',False),info.get('crop_padding',DEFAULT_CROP_PADDING)) for _,info in models}
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
                views = {}
                for color,padding in variants:
                    region = crop(image,row['box'],padding)
                    valid = region.size and min(region.shape[:2])>=5
                    views[color,padding] = encode(region,color) if valid else None
                pending.append((row['detection_id'],views))
                if len(pending)==32:
                    flush()
        if pending:
            flush()
    finally:
        capture.release()
    return output


def apply_consensus(observations,predictions):
    rows = [{**row,'identity':predictions[row['detection_id']]} for row in observations]
    for track in associate(rows):
        stable = summarize_track(track)
        for row in track:
            identity = row['identity']
            if identity and identity['frame_accepted'] and not stable:
                identity.update(accepted=False,reason='insufficient_track_agreement')
    for row in rows:
        if row.get('duplicate_of') and row['identity']:
            row['identity'].update(accepted=False,reason='duplicate_proposal')


def classify_observations(source, observations, model, info):
    predictions, = predict_video(source,observations,[(model,info)])
    apply_consensus(observations,predictions)
    return predictions


def associate(rows, *, max_gap=MAX_TRACK_GAP, duplicate_iou=.72, duplicate_coverage=.88):
    active = {}
    track = 0
    grouped = {}
    history = defaultdict(Counter)
    by_time = {}
    for row in rows:
        by_time.setdefault(row['time'],[]).append(row)

    def matches(a,b):
        previous = history.get(b.get('track_id'),{})
        current_label = label(a)
        if incompatible(a,b) or (current_label and len(previous)==1 and previous.get(current_label,0)==0
                                 and sum(previous.values())>=2):
            return False
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

    def area(row):
        x,y,r,d = row['box']
        return max(0,r-x)*max(0,d-y)

    def label(row):
        return label_evidence(row.get('identity'))

    def incompatible(a,b):
        first,second = label(a),label(b)
        return bool(first and second and first!=second)

    def duplicate(a,b):
        if incompatible(a,b):
            return False
        x,y,r,d = a['box']
        u,v,s,t = b['box']
        intersection = max(0,min(r,s)-max(x,u))*max(0,min(d,t)-max(y,v))
        smaller, larger = sorted((area(a),area(b)))
        if smaller<=0:
            return False
        overlap = intersection/(smaller+larger-intersection)
        width,height = max(r-x,s-u),max(d-y,t-v)
        near = abs(x+r-u-s)<=.40*width and abs(y+d-v-t)<=.65*height
        return overlap>=duplicate_iou or (near and intersection>=duplicate_coverage*smaller and intersection>=.36*larger)

    def quality(row):
        identity = row.get('identity') or {}
        accepted = identity.get('frame_accepted',identity.get('accepted',False))
        # Prefer a tight, valid crop over a larger box containing substantial background.
        return (not accepted,-int(identity.get('confidence',0)*50),area(row),
                -(row.get('confidence') or 0))

    for time, rows_at_time in sorted(by_time.items()):
        current = []
        for row in sorted(rows_at_time,key=quality):
            row.pop('duplicate_of',None)
            row.pop('track_id',None)
            row.pop('track_break',None)
            row.pop('track_end',None)
            row.pop('interrupted_from',None)
            matches_in_frame = [item for item in current if duplicate(row,item)]
            if len(matches_in_frame)==1:
                row['duplicate_of'] = matches_in_frame[0].get('detection_id')
            else:
                current.append(row)

        for identifier, row in list(active.items()):
            if time-row['time']>max_gap:
                row['track_end'] = 'gap_expired'
                del active[identifier]
        links = {id(row):[item for item in active.values() if matches(row,item)] for row in current}
        blocked = set()
        for row in current:
            options = links[id(row)]
            if len(options)==1 and sum(options[0] in links[id(other)] for other in current)==1:
                row['track_id'] = options[0]['track_id']
            else:
                if options:
                    row['track_break'] = 'association_ambiguous'
                    row['interrupted_from'] = [item['track_id'] for item in options]
                    blocked.update(item['track_id'] for item in options)
                track += 1
                row['track_id'] = f'V-{track}'
            grouped.setdefault(row['track_id'],[]).append(row)
            if label(row):
                history[row['track_id']][label(row)] += 1
        for identifier in blocked:
            ended = active.pop(identifier,None)
            if ended:
                ended['track_end'] = 'association_ambiguous'
        active.update((row['track_id'],row) for row in current)
    return grouped.values()


def summarize_track(track):
    votes = defaultdict(list)
    for row in track:
        identity = row.get('identity') or {}
        if identity.get('reason')!='classifier_disagreement' and identity.get('frame_accepted',identity.get('accepted',False)):
            votes[identity['class_id']].append(identity)
    if not votes:
        return None
    ranked = sorted(votes.values(),key=lambda entries:sum(item['confidence'] for item in entries),reverse=True)
    best = ranked[0]
    if len(best)<2 or len(ranked)>1:
        return None
    return {**best[0],'accepted':True,'confidence':sum(item['confidence'] for item in best)/len(best),
            'supporting_frames':len(best)}


def describe_tracks(rows, video, reviews):
    """Build review and playback state from immutable proposals and predictions."""
    tracks = list(associate(rows))
    by_id = {row['detection_id']:row for row in rows}
    reviewed = {item['detection_id'] for item in reviews if item['decision']=='confirmed'}
    ignored = {item['detection_id'] for item in reviews if item.get('review_action')=='non_instrument'}
    reviewed |= {by_id[key]['duplicate_of'] for key in reviewed if key in by_id and by_id[key].get('duplicate_of')}
    sample_period = max(1,round(video['fps']/5))/video['fps']
    descriptors = []
    aliases = defaultdict(list)
    for row in rows:
        if key:=row.get('duplicate_of'):
            aliases[key].append(row['detection_id'])
            row.update(instrument_status='duplicate',reason_code='duplicate_proposal')

    stable = {track[0]['track_id']:summarize_track(track) for track in tracks}
    for track in tracks:
        identity = stable[track[0]['track_id']]
        has_tool_evidence = identity or any(
            row['detection_id'] in reviewed or label_evidence(row.get('identity')) for row in track)
        context_rows = [{'identity':{**evidence,'frame_accepted':evidence['accepted']}}
                        for row in track if (evidence:=(row.get('identity') or {}).get('context'))]
        context = summarize_track(context_rows) if not has_tool_evidence else None
        support = [row['time'] for row in track
                   if identity and label_evidence(row.get('identity'))==identity['class_id']]
        interruption = (track[0].get('track_break') if any(
            stable.get(identifier) for identifier in track[0].get('interrupted_from',[])) else None)
        first,last = track[0]['time'],track[-1]['time']
        termination = track[-1].get('track_end')
        if not termination:
            termination = 'recording_ended' if video['duration']-last<=MAX_TRACK_GAP else 'gap_expired'
        for row in track:
            prediction = row.get('identity')
            if row['detection_id'] in ignored:
                row.update(instrument_status='non_instrument',reason_code='reviewer_marked_non_instrument')
            elif context:
                row.update(instrument_status='context_detected',reason_code='context_model_agreement',context_identity=context)
            elif support and min(abs(row['time']-at) for at in support)<=MAX_TRACK_GAP:
                row.update(instrument_status='ai_predicted',reason_code='consistent_identity',track_identity=identity)
            elif interruption and row is track[0]:
                row.update(instrument_status='track_interrupted',reason_code=interruption)
            elif not prediction or len(track)==1:
                row.update(instrument_status='unclassified_candidate',
                           reason_code=(prediction or {}).get('reason') or 'insufficient_track_support')
            else:
                row.update(instrument_status='type_uncertain',
                           reason_code=prediction.get('reason') or 'insufficient_track_support')

        status = ('non_instrument' if all(row['detection_id'] in ignored for row in track) else
                  'context_detected' if context else 'ai_predicted' if identity else
                  'track_interrupted' if interruption else 'type_uncertain' if len(track)>1 and
                  any(row.get('identity') for row in track) else 'unclassified_candidate')
        canonical = [row['detection_id'] for row in track]
        lapses = [{'start':a['time'],'end':b['time'],'status':'track_interrupted','reason':'missed_sample'}
                  for a,b in zip(track,track[1:]) if b['time']-a['time']>sample_period*1.5]
        descriptors.append(dict(
            track_id=track[0]['track_id'],start=first,end=last,samples=len(track),duration=last-first,
            identity=identity,context_identity=context,status=status,termination_reason=termination,
            termination_status='track_interrupted' if termination=='association_ambiguous' else
            'track_lost' if termination=='gap_expired' else 'recording_ended',
            gaps=lapses,canonical_detection_ids=canonical,
            raw_detection_ids=[key for item in canonical for key in [item,*aliases[item]]],
            evidence=[dict(detection_id=row['detection_id'],time=row['time'], frame=row['frame'],
                           status=row['instrument_status'],reason=row['reason_code'],
                           predicted_class=(row.get('identity') or {}).get('class_id'),
                           score=(row.get('identity') or {}).get('confidence')) for row in track],
        ))
    diagnostics = {
        'raw_proposals':len(rows),'canonical_proposals':sum(len(track) for track in tracks),
        'duplicate_proposals':sum(row.get('instrument_status')=='duplicate' for row in rows),
        'proposal_statuses':dict(Counter(row['instrument_status'] for row in rows)),
        'reason_codes':dict(Counter(row['reason_code'] for row in rows)),
        'tracks':len(tracks),'one_frame_tracks':sum(len(track)==1 for track in tracks),
        'track_statuses':dict(Counter(item['status'] for item in descriptors)),
        'interrupted_tracks':sum(bool(item['gaps']) or item['termination_status']=='track_interrupted' for item in descriptors),
        'reviews':dict(Counter(row.get('review_action',row['decision']) for row in reviews)),
    }
    return descriptors, diagnostics


def label_evidence(identity):
    if identity and identity.get('reason')!='classifier_disagreement' and identity.get('frame_accepted',identity.get('accepted',False)):
        return identity['class_id']
    return None
