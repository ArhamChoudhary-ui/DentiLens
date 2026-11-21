"""Source-grouped archive manifests and validation for the 4.3 benchmark."""
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from instrument_taxonomy import LABELS, TAXONOMY, canonical

SPLITS = {'train', 'valid', 'test'}
VERSION = 'dentalguard-4.3-v1'
SOURCE = 'dentalguard'
SOURCE_FRAME = re.compile(r'^(\d+)_frame_(\d+)')
SOURCES = yaml.safe_load((ROOT/'configs/dataset_sources.yaml').read_text())['sources']


def checksum(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')


def image_from_archive(archive, member):
    payload = archive.read(member)
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Invalid image: {member}')
    return payload, image


def valid_box(box):
    x, y, r, b = box
    return 0 <= x < r <= 1 and 0 <= y < b <= 1


def parse_labels(content, names, source=SOURCE):
    annotations = []
    for line in content.splitlines():
        values = line.split()
        if len(values) != 9:
            raise ValueError('Expected a class and four normalized vertices.')
        class_index = int(values[0])
        if class_index < 0 or class_index >= len(names):
            raise ValueError('Invalid source class index.')
        original = names[class_index]
        points = np.array(values[1:], dtype=float).reshape(4, 2)
        if not np.isfinite(points).all() or np.any((points < 0) | (points > 1)):
            raise ValueError('Invalid polygon coordinates.')
        box = [*points.min(axis=0), *points.max(axis=0)]
        if not valid_box(box) or cv2.contourArea(points.astype('float32')) <= 0:
            raise ValueError('Empty annotation.')
        annotations.append(dict(original_class=original, label=canonical(original, source),
                                box=[float(v) for v in box], polygon=points.tolist(),
                                occlusion=None, visibility=None, status='source_labeled', corrections=[]))
    return annotations


def image_signature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    values = np.packbits(resized[:, 1:] > resized[:, :-1])
    return values.tobytes().hex()


def leakage(rows, max_distance=4):
    groups = defaultdict(set)
    patients = defaultdict(set)
    exact = defaultdict(set)
    by_split = defaultdict(dict)
    for row in rows:
        groups[row['recording']].add(row['split'])
        if row.get('patient'):
            patients[row['patient']].add(row['split'])
        exact[row['sha256']].add(row['split'])
        by_split[row['split']].setdefault(int(row['dhash'], 16), row['id'])
    near = []
    splits = sorted(by_split)
    for i, a in enumerate(splits):
        for b in splits[i+1:]:
            for lhs, source_id in by_split[a].items():
                for rhs, other_id in by_split[b].items():
                    if (lhs ^ rhs).bit_count() <= max_distance:
                        near.append([source_id, other_id])
    return dict(recording_conflicts=sorted(k for k,v in groups.items() if len(v)>1),
                patient_conflicts=sorted(k for k,v in patients.items() if len(v)>1),
                exact_conflicts=sorted(k for k,v in exact.items() if len(v)>1),
                near_duplicate_pairs=near, dhash_max_distance=max_distance)


def validate(metadata, frames, crops):
    if metadata['version'] != VERSION or metadata['taxonomy_version'] != TAXONOMY['version']:
        raise ValueError('Unsupported manifest version.')
    source = metadata['source']
    if source not in SOURCES or any(not SOURCES[source].get(k) for k in ('url','release','license','evidence')):
        raise ValueError('Source provenance is incomplete.')
    if metadata.get('provenance') != SOURCES[source] or not SOURCES[source]['local_benchmark']:
        raise ValueError('The manifest source is not approved in the benchmark registry.')
    ids = {}
    origins = set()
    hashes = set()
    for frame in frames:
        origin = (frame['recording'], frame['frame'])
        if frame['id'] in ids or origin in origins or frame['sha256'] in hashes:
            raise ValueError('Duplicate frame identity or image.')
        ids[frame['id']] = frame
        origins.add(origin)
        hashes.add(frame['sha256'])
        if not frame['recording'] or frame['source'] != source or frame['split'] not in SPLITS:
            raise ValueError('Invalid frame provenance or split.')
        if metadata['recording_splits'].get(frame['recording']) != frame['split']:
            raise ValueError('Recording assignment mismatch.')
        if min(frame['width'],frame['height']) < 1 or frame['frame'] < 0 or (not frame['annotations'] and not frame.get('verified_negative')):
            raise ValueError('Invalid dimensions, frame number, or empty labels.')
        if frame.get('verified_negative') and any(a['label']!='non_instrument' for a in frame['annotations']):
            raise ValueError('Verified negative frame includes an instrument.')
        boxes = {}
        for annotation in frame['annotations']:
            label, box = annotation['label'], annotation['box']
            if label not in LABELS or not valid_box(box):
                raise ValueError('Invalid class or bounding box.')
            if label != canonical(annotation['original_class'],source):
                raise ValueError('Original class mapping mismatch.')
            key = tuple(box)
            if key in boxes:
                raise ValueError('Duplicate or conflicting identical boxes.')
            boxes[key] = label
    crop_ids = set()
    for row in crops:
        parent = ids.get(row['parent'])
        if row['id'] in crop_ids or parent is None or row['split'] != parent['split']:
            raise ValueError('Duplicate crop or invalid parent assignment.')
        crop_ids.add(row['id'])
        index = row['annotation_index']
        if not 0 <= index < len(parent['annotations']):
            raise ValueError('Invalid crop annotation index.')
        annotation = parent['annotations'][index]
        if any(row[k] != annotation[k] for k in ('label','box','original_class')):
            raise ValueError('Crop and parent annotation differ.')
    return leakage(frames)


def load_dataset(root):
    return (read_json(root/'manifests/dataset.json'), read_json(root/'intraoral_detection/frames.json'),
            read_json(root/'instrument_classification/crops.json'))


def summary(rows, label_key=None):
    counts = defaultdict(Counter)
    for row in rows:
        labels = [row[label_key]] if label_key else [annotation['label'] for annotation in row['annotations']]
        counts[row['split']].update(labels)
    return {split:dict(counts[split]) for split in sorted(counts)}
