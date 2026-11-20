"""Convert the supplied Roboflow archive to one-class tooth detection data.

Only enclosing boxes are used: this export's polygons are not assumed to be
anatomically accurate segmentation masks. Original archive splits are retained.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import tempfile
import zipfile

import yaml


def prepare(archive, destination):
    archive, destination = Path(archive), Path(destination).resolve()
    if destination.exists():
        raise ValueError('Choose a new dataset destination')
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts, class_counts = {}, Counter()
    source_splits, image_splits = defaultdict(set), defaultdict(set)
    polygon_shapes = Counter()
    with zipfile.ZipFile(archive) as source, tempfile.TemporaryDirectory(
            prefix='.intraoral-', dir=destination.parent) as temporary:
        stage = Path(temporary)
        names = set(source.namelist())
        config = yaml.safe_load(source.read('data.yaml'))
        for split in ('train', 'valid', 'test'):
            images = sorted(n for n in names if n.startswith(f'{split}/images/')
                            and Path(n).suffix.lower() in ('.jpg', '.jpeg', '.png'))
            if not images:
                raise ValueError(f'Missing {split} split')
            (stage / split / 'images').mkdir(parents=True)
            (stage / split / 'labels').mkdir()
            boxes_count = 0
            for image in images:
                basename = Path(image).name
                label_name = f'{split}/labels/{Path(image).stem}.txt'
                if label_name not in names:
                    raise ValueError(f'Missing label for {image}')
                content = source.read(image)
                image_splits[hashlib.sha256(content).hexdigest()].add(split)
                source_splits[Path(image).stem.split('.rf.')[0]].add(split)
                boxes = []
                for row in source.read(label_name).decode().splitlines():
                    values = list(map(float, row.split()))
                    if not values or not all(math.isfinite(v) for v in values):
                        raise ValueError(f'Invalid annotation in {label_name}')
                    class_id = values[0]
                    if not class_id.is_integer() or not 0 <= class_id < config['nc']:
                        raise ValueError(f'Invalid class in {label_name}')
                    coordinates = values[1:]
                    if not all(0 <= v <= 1 for v in coordinates):
                        raise ValueError(f'Out-of-bounds annotation in {label_name}')
                    if len(values) == 5:
                        x, y, width, height = coordinates
                        polygon_shapes['box'] += 1
                    elif len(values) >= 7 and len(values) % 2 == 1:
                        xs, ys = coordinates[::2], coordinates[1::2]
                        x, y = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
                        width, height = max(xs)-min(xs), max(ys)-min(ys)
                        polygon_shapes['polygon_vertices_' + str(len(xs))] += 1
                    else:
                        raise ValueError(f'Unsupported annotation in {label_name}')
                    if width <= 0 or height <= 0:
                        raise ValueError(f'Empty box in {label_name}')
                    boxes.append('0 ' + ' '.join(f'{v:.8f}' for v in (x, y, width, height)))
                    class_counts[config['names'][int(class_id)]] += 1
                (stage / split / 'images' / basename).write_bytes(content)
                (stage / label_name).write_text('\n'.join(boxes) + ('\n' if boxes else ''))
                boxes_count += len(boxes)
            counts[split] = {'images': len(images), 'boxes': boxes_count}
        if any(len(splits) > 1 for splits in image_splits.values()):
            raise ValueError('Identical images occur across dataset splits')
        if any(len(splits) > 1 for splits in source_splits.values()):
            raise ValueError('Source basenames occur across dataset splits')
        for name in ('README.dataset.txt', 'README.roboflow.txt'):
            if name in names:
                (stage / name).write_bytes(source.read(name))
        dataset = {'path': str(destination), 'train': 'train/images',
                   'val': 'valid/images', 'test': 'test/images', 'names': {0: 'tooth'}}
        (stage / 'data.yaml').write_text(yaml.safe_dump(dataset, sort_keys=False))
        report = {'archive_sha256': hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest(),
                  'source': config.get('roboflow', {}), 'splits': counts,
                  'source_class_instances': dict(class_counts),
                  'annotation_shapes': dict(polygon_shapes),
                  'output_classes': ['tooth'],
                  'label_policy': 'All source tooth classes, including 327 and 56, collapsed to tooth; no FDI numbering claim.',
                  'cross_split_identical_hashes': 0, 'cross_split_source_basenames': 0,
                  'limitations': ['No patient identifiers available to audit patient overlap across splits.',
                                  'Near-duplicate images with different basenames are not excluded by hash checks.',
                                  'Enclosing boxes do not establish accurate tooth-boundary masks.']}
        (stage / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
        stage.rename(destination)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.destination), indent=2))
