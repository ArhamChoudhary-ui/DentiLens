"""Load source-separated instrument crops from the DentalGuard video frames."""
from collections import defaultdict
from pathlib import Path
import random
import re
from zipfile import ZipFile

import cv2
import numpy as np
import yaml

from instrument_identity import crop, encode


def dentalguard(path, context=False, *, color=True):
    groups = defaultdict(lambda: defaultdict(list))
    targets = {'Explorer No.5':'dental_explorer','Mouth mirror':'dental_mirror'}
    negatives = {'hand_gloved','hand_ungloved','mouth_open'}
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        labels = yaml.safe_load(archive.read('data.yaml'))['names']
        if not set(targets) | negatives <= set(labels.values()):
            raise ValueError('Required DentalGuard labels are missing.')
        for file in sorted(names):
            source = re.match(r'(\d+)_frame_',Path(file).name)
            if not source or '/labels/' not in file or not file.endswith('.txt'):
                continue
            image_path = file.replace('/labels/','/images/').removesuffix('.txt')+'.jpg'
            if image_path not in names:
                continue
            boxes = []
            for row in archive.read(file).decode().splitlines():
                parts = row.split()
                if len(parts)!=9:
                    continue
                coords = np.asarray(parts[1:],dtype=float).reshape(4,2).clip(0,1)
                left,top = coords.min(axis=0)
                right,bottom = coords.max(axis=0)
                if left<right and top<bottom:
                    boxes.append((labels[int(parts[0])],(left,top,right,bottom)))
            tools = [box for label,box in boxes if label not in negatives]
            for label,box in boxes:
                if label in targets:
                    name = 'unknown' if context else targets[label]
                elif label in negatives and not any(overlap(box,tool) for tool in tools):
                    name = label if context else 'unknown'
                else:
                    continue
                groups[source[1]][name].append((image_path,box))
        sources = sorted(groups)
        if len(sources)<12:
            raise ValueError('Need at least 12 video groups for a source-separated check.')
        rng = random.Random(44)
        rng.shuffle(sources)
        splits = {'guard_test':sources[:4], 'guard_valid':sources[4:8], 'train':sources[8:]}
        data = {name:[] for name in splits}
        origins = {}
        for split, keys in splits.items():
            for source in keys:
                origins[source] = split
                for label, specs in sorted(groups[source].items()):
                    limit = 28 if label=='unknown' else 30
                    for image_path,box in rng.sample(specs,min(limit,len(specs))):
                        image=cv2.imdecode(np.frombuffer(archive.read(image_path),np.uint8),cv2.IMREAD_COLOR)
                        region=crop(image,box)
                        if region.size and min(region.shape[:2])>=5:
                            data[split].append((encode(region,color=color),label))
    return data, origins


def overlap(a,b):
    left,top,right,bottom = a
    x,y,r,d = b
    intersection = max(0,min(right,r)-max(left,x))*max(0,min(bottom,d)-max(top,y))
    return intersection > .1*(r-x)*(d-y)
