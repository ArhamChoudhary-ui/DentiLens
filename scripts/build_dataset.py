"""Build source-bound detection and crop manifests without copying archive media."""
import argparse
from collections import Counter
from hashlib import sha256
from zipfile import ZipFile

from instrument_dataset import (ROOT, SOURCE, SOURCES, SOURCE_FRAME, VERSION, TAXONOMY, Path,
                                checksum, image_from_archive, image_signature, leakage, parse_labels, read_json, summary,
                                validate, write_json, yaml)


def intersects(box, other):
    return min(box[2],other[2]) > max(box[0],other[0]) and min(box[3],other[3]) > max(box[1],other[1])


def build(path, splits):
    counts = Counter()
    excluded = []
    frames, crops = [], []
    seen_origins, seen_hashes = set(), set()
    source = SOURCES[SOURCE]
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        notice = archive.read(source['evidence']).decode()
        if source['license'] not in notice or source['url'] not in notice:
            raise ValueError('Archive license or source does not match the registry.')
        label_names = yaml.safe_load(archive.read('data.yaml'))['names']
        members = sorted(name for name in names if '/images/' in name and name.endswith('.jpg'))
        counts['archive_images'] = len(members)
        for name in members:
            origin = SOURCE_FRAME.match(Path(name).name)
            if not origin or origin[1] not in splits:
                counts['without_verified_source_split'] += 1
                continue
            recording, index = origin[1], int(origin[2])
            split = splits[recording]
            try:
                label_file = name.replace('/images/','/labels/').removesuffix('.jpg')+'.txt'
                annotations = parse_labels(archive.read(label_file).decode(), label_names)
                if not annotations:
                    raise ValueError('Unverified empty annotation')
                if len({tuple(a['box']) for a in annotations}) != len(annotations):
                    raise ValueError('Duplicate or conflicting boxes')
                data, image = image_from_archive(archive, name)
                digest = sha256(data).hexdigest()
                if (recording,index) in seen_origins or digest in seen_hashes:
                    raise ValueError('Duplicate origin or exact image')
            except (KeyError,ValueError) as exc:
                excluded.append(dict(member=name, reason=str(exc)))
                continue
            height,width = image.shape[:2]
            identifier = f'{recording}-{index}'
            frames.append(dict(id=identifier, recording=recording, patient=None, source=SOURCE, frame=index,
                               timestamp=None, split=split, member=name, width=width, height=height,
                               sha256=digest, dhash=image_signature(image), annotations=annotations))
            seen_origins.add((recording,index))
            seen_hashes.add(digest)
            tools = [a['box'] for a in annotations if a['label']!='non_instrument']
            for position, annotation in enumerate(annotations):
                box = annotation['box']
                dx,dy = (box[2]-box[0])*.1, (box[3]-box[1])*.1
                padded = [box[0]-dx,box[1]-dy,box[2]+dx,box[3]+dy]
                if annotation['label']=='non_instrument' and any(intersects(padded,tool) for tool in tools):
                    counts['mixed_context_crops_excluded'] += 1
                    continue
                if (box[2]-box[0])*width < 5 or (box[3]-box[1])*height < 5:
                    counts['tiny_crops_excluded'] += 1
                    continue
                crops.append(dict(id=f'{identifier}-{position}',parent=identifier,annotation_index=position,
                                  split=split, **{k:annotation[k] for k in ('label','box','original_class')}))
    before_quarantine = leakage(frames)
    by_id = {row['id']: row for row in frames}
    priority = {'test':0,'valid':1,'train':2}
    quarantine = set()
    for pair in before_quarantine['near_duplicate_pairs']:
        lower_priority = max((by_id[key] for key in pair),key=lambda row:priority[row['split']])
        quarantine.add((lower_priority['split'],lower_priority['dhash']))
    keep = []
    for row in frames:
        if (row['split'],row['dhash']) in quarantine:
            excluded.append(dict(member=row['member'],reason='Cross-split perceptual similarity quarantine'))
            counts['near_duplicate_frames_quarantined'] += 1
        else:
            keep.append(row)
    frames = keep
    parent_ids = {row['id'] for row in frames}
    crops = [row for row in crops if row['parent'] in parent_ids]
    metadata = dict(version=VERSION, taxonomy_version=TAXONOMY['version'], annotation_version=1,
                    creation_date='2026-11-21', source=SOURCE, archive_sha256=checksum(path), provenance=source,
                    recording_splits=splits, archive_classes=label_names, exclusions=excluded,
                    preprocessing={'detection':'source-resolution normalized AABB from original OBB',
                                   'classification':'production crop, 10% padding, aspect-fit 96px'},
                    counts=dict(counts), pre_quarantine_leakage=before_quarantine, detection_counts=summary(frames), classification_counts=summary(crops,'label'))
    metadata['leakage'] = validate(metadata, frames, crops)
    return metadata,frames,crops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    splits = read_json(ROOT/'datasets/manifests/dentalguard-splits.json')
    metadata, frames, crops = build(args.archive, splits)
    args.output.mkdir(parents=True, exist_ok=False)
    for sub in ('manifests','intraoral_detection','instrument_classification'):
        (args.output/sub).mkdir()
    write_json(args.output/'manifests/dataset.json',metadata)
    write_json(args.output/'intraoral_detection/frames.json',frames)
    write_json(args.output/'instrument_classification/crops.json',crops)
    print(f'{len(frames)} frames, {len(crops)} crops, {len(metadata["exclusions"])} excluded numbered files')
    print(metadata['counts'])
    print({key:len(values) for key,values in metadata['leakage'].items() if isinstance(values,list)})


if __name__=='__main__':
    main()
