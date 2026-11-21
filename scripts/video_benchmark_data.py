"""Local, purpose-gated enrollment and original-frame annotation."""
from datetime import date
from hashlib import file_digest
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import cv2
import yaml

from instrument_dataset import ROOT, LABELS, valid_box, write_json

PERMISSIONS = yaml.safe_load((ROOT/'configs/video_permissions.yaml').read_text())
SCHEMA = yaml.safe_load((ROOT/'configs/video_annotation_schema.yaml').read_text())
CANDIDATES = set(LABELS)-{'non_instrument'}


def require(source, *actions):
    if source['status'] != 'authorized':
        raise ValueError('Source is quarantined or authorization is inactive.')
    expiry = source.get('expires')
    if expiry and date.fromisoformat(expiry) < date.today():
        raise ValueError('Source authorization has expired.')
    if not source.get('rights_reference') or not source.get('rights_reviewer') or not source.get('license'):
        raise ValueError('Missing rights provenance.')
    if not set(actions) <= set(PERMISSIONS['actions']):
        raise ValueError('Invalid permission purpose.')
    declared = source.get('permissions',{})
    restricted = set(actions) - {key for key,value in declared.items() if value is True}
    if restricted:
        raise ValueError('Missing permission: '+', '.join(sorted(restricted)))
    if source.get('identifiable') is not False:
        consent = source.get('consent',{})
        if not consent.get('reference') or any(consent.get('permissions',{}).get(key) is not True for key in actions):
            raise ValueError('Explicit consent is required for the requested purpose.')


def inspect_video(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('Cannot read source video.')
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps<=0 or min(frames,width,height)<=0:
            raise ValueError('Invalid source video metadata.')
        return dict(fps=fps,frames=frames,width=width,height=height,duration=frames/fps)
    finally:
        capture.release()


def digest(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def register(record):
    required = ('id','name','reference','category','recording_group','split','status')
    if any(not record.get(key) for key in required) or not re.fullmatch('[a-zA-Z0-9_-]+',record['id']):
        raise ValueError('Missing provenance or invalid recording ID.')
    if record['category'] not in PERMISSIONS['categories'] or record['status'] not in PERMISSIONS['statuses']:
        raise ValueError('Invalid source category or permission state.')
    if record['split'] not in {'train','valid','test','quarantine'}:
        raise ValueError('Invalid dataset assignment.')
    enrolled = dict(record)
    enrolled.pop('video',None)
    enrolled.pop('sha256',None)
    if record['status']!='authorized':
        enrolled['split']='quarantine'
        return enrolled
    require(record,'view','store')
    if record['split']=='test' and record.get('previously_observed') is not False:
        raise ValueError('Final test sources need an explicit untouched-source attestation.')
    path=Path(record['path']).resolve()
    enrolled.update(path=str(path),video=inspect_video(path),sha256=digest(path))
    return enrolled


def source_path(record,*actions):
    require(record,'view','store',*actions)
    if record['split']=='quarantine' or record.get('cfr_verified') is not True:
        raise ValueError('Source needs an eligible split and verified constant FPS.')
    path=Path(record['path'])
    if digest(path)!=record['sha256']:
        raise ValueError('Source video changed after registration.')
    return path


def check_isolation(records):
    for key in ('recording_group','patient_group','sha256'):
        groups={}
        for record in records:
            value=record.get(key)
            if value is None or record['split']=='quarantine':
                continue
            if groups.setdefault(value,record['split'])!=record['split']:
                raise ValueError(f'Cross-split source overlap: {key}')
    if len({record['id'] for record in records})!=len(records):
        raise ValueError('Duplicate recording ID.')
    hashes=[record['sha256'] for record in records if record.get('sha256') and record['split']!='quarantine']
    if len(set(hashes))!=len(hashes):
        raise ValueError('Duplicate video content under multiple registrations.')


def sampling(video):
    return range(0,video['frames'],max(1,round(video['fps']/5)))


def iter_frames(path,video):
    scheduled=sampling(video)
    capture=cv2.VideoCapture(str(path))
    try:
        for frame in range(video['frames']):
            if not capture.grab():
                raise ValueError(f'Could not read source frame {frame}.')
            if frame in scheduled:
                ok,image=capture.retrieve()
                if not ok or image.shape[:2]!=(video['height'],video['width']):
                    raise ValueError('Could not decode frame at registered resolution.')
                yield frame,image
    finally:
        capture.release()


def extract(record,output):
    path=source_path(record,'extract','annotate')
    output.mkdir(parents=True,exist_ok=False)
    frames=[]
    for frame,image in iter_frames(path,record['video']):
        filename=f'{frame:08d}.jpg'
        if not cv2.imwrite(str(output/filename),image):
            raise ValueError('Could not write benchmark sample.')
        frames.append(dict(frame=frame,time=frame/record['video']['fps'],file=filename))
    write_json(output/'frames.json',dict(recording=record['id'],sha256=record['sha256'],frames=frames))


def read_cvat(path,video):
    """Import CVAT video-track XML. Interpolate visible keyframes on the native grid."""
    tree=ET.parse(path).getroot()
    expected=tree.find('./meta/task/original_size')
    if expected is not None and (int(expected.findtext('width')),int(expected.findtext('height'))) != (
            video['width'],video['height']):
        raise ValueError('CVAT video dimensions do not match the recording.')
    if tree.findall('./image'):
        raise ValueError('Export CVAT video tracks with original source-frame indices.')
    result=[]
    seen=set()
    for track in tree.findall('./track'):
        identity=track.attrib['id']
        label=track.attrib['label']
        if identity in seen or label not in CANDIDATES:
            raise ValueError('Duplicate track or invalid instrument class.')
        seen.add(identity)
        keys=[]
        for box in track.findall('box'):
            frame=int(box.attrib['frame'])
            if not 0<=frame<video['frames']:
                raise ValueError('CVAT frame outside source.')
            outside=box.get('outside')=='1'
            visibility=next((attr.text for attr in box.findall('attribute') if attr.get('name')=='visibility'),
                            'outside' if outside else 'partial' if box.get('occluded')=='1' else 'visible')
            if (visibility not in SCHEMA['visibility']
                    or outside!=(visibility in {'outside','invisible'})):
                raise ValueError('Visibility must agree with the CVAT outside flag.')
            coords=[float(box.attrib[field])/axis for field,axis in zip(
                ('xtl','ytl','xbr','ybr'),(video['width'],video['height'])*2)]
            if not outside and not valid_box(coords):
                raise ValueError('Invalid CVAT box.')
            keys.append((frame, None if outside else coords, visibility))
        if any(a[0]>=b[0] for a,b in zip(keys,keys[1:])) or not keys:
            raise ValueError('Empty track or unordered/duplicate CVAT frame.')
        for i,(start,box,visibility) in enumerate(keys):
            end,next_box,_ = keys[i+1] if i+1<len(keys) else (video['frames'],None,None)
            cadence=sampling(video).step
            first=((start+cadence-1)//cadence)*cadence
            for frame in range(first,end,cadence):
                coords=box
                if box is not None and next_box is not None and frame!=start:
                    ratio=(frame-start)/(end-start)
                    coords=[a+(b-a)*ratio for a,b in zip(box,next_box)]
                result.append(dict(frame=frame,time=frame/video['fps'],track_id=identity,
                                   label=label,box=coords,visibility=visibility,
                                   occluded=visibility in {'partial','mostly_occluded','invisible'}))
    return result


def reviewed_annotations(record,xml,review):
    source_path(record,'extract','annotate')
    video=record['video']
    if (review.get('review_state')!='verified' or not review.get('reviewer')
            or review.get('source_sha256')!=record['sha256']
            or review.get('annotation_sha256')!=digest(xml)
            or review.get('frames')!=list(sampling(video))):
        raise ValueError('Review must attest to every scheduled sample and bind both file hashes.')
    rows=read_cvat(xml,video)
    events=[]
    for identity in sorted({row['track_id'] for row in rows}):
        track=[row for row in rows if row['track_id']==identity]
        visible=[row for row in track if row['box'] is not None]
        if visible:
            step=sampling(video).step
            last=visible[-1]
            exit_frame=last['frame']+step
            exit_sample=next((row for row in track if row['frame']==exit_frame),None)
            outside=exit_sample is not None and exit_sample['visibility']=='outside'
            events.append(dict(track_id=identity,label=visible[0]['label'],entry=visible[0]['time'],
                               last_seen=last['time'],exit=exit_sample['time'] if outside else None))
    return dict(recording=record['id'],version=SCHEMA['version'],frames=list(sampling(video)),
                objects=rows,events=events,review=review)
