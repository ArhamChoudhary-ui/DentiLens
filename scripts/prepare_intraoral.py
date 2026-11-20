"""Prepare dental image archives for box detection.

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
from PIL import Image
from io import BytesIO


TOOTH_TYPES = ['central incisor', 'lateral incisor', 'canine', 'first premolar',
               'second premolar', 'first molar', 'second molar', 'third molar']


FDI_CLASSES = [str(10 * quadrant + tooth) for quadrant in range(1, 5) for tooth in range(1, 9)]


def prepare(archive, destination, tooth_types=False, task=None):
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
        yaml_path = 'Cavity Dataset/data.yaml' if task == 'cavity' else 'data.yaml'
        prefix = str(Path(yaml_path).parent) + '/' if task == 'cavity' else ''
        config = yaml.safe_load(source.read(yaml_path))
        source_names = config['names']
        output_classes = {'fdi': FDI_CLASSES, 'cavity': ['possible cavity', 'normal appearance']}.get(
            task, TOOTH_TYPES if tooth_types else ['tooth'])
        if task == 'cavity' and source_names != ['cavity', 'normal']:
            raise ValueError('Expected cavity and normal source classes')
        omitted = Counter()
        for split in ('train', 'valid', 'test'):
            images = sorted(n for n in names if n.startswith(f'{prefix}{split}/images/')
                            and Path(n).suffix.lower() in ('.jpg', '.jpeg', '.png'))
            if not images:
                raise ValueError(f'Missing {split} split')
            (stage / split / 'images').mkdir(parents=True)
            (stage / split / 'labels').mkdir()
            boxes_count = 0
            for image in images:
                basename = Path(image).name
                label_name = f'{split}/labels/{Path(image).stem}.txt'
                source_label_path = prefix + (label_name.replace('/labels/', '/labelTxt/')
                                              if task == 'cavity' else label_name)
                if source_label_path not in names:
                    raise ValueError(f'Missing label for {image}')
                content = source.read(image)
                image_splits[hashlib.sha256(content).hexdigest()].add(split)
                source_splits[Path(image).stem.split('.rf.')[0]].add(split)
                boxes = []
                if task == 'cavity':
                    with Image.open(BytesIO(content)) as image_file:
                        image_width, image_height = image_file.size
                for row in source.read(source_label_path).decode().splitlines():
                    if not row.strip():
                        continue
                    columns = row.split()
                    if task == 'cavity':
                        class_index = source_names.index(columns[8])
                        values = [float(class_index)] + [float(value) / dimension for value, dimension in
                            zip(columns[:8], [image_width, image_height] * 4)]
                        if any(not math.isfinite(v) or v < -1e-6 or v > 1.000001 for v in values[1:]):
                            omitted['invalid_cavity_box'] += 1
                            continue
                        values = [values[0]] + [min(1, max(0, v)) for v in values[1:]]
                    else:
                        values = list(map(float, columns))
                    if not values or not all(math.isfinite(v) for v in values):
                        raise ValueError(f'Invalid annotation in {label_name}')
                    class_id = values[0]
                    if not class_id.is_integer() or not 0 <= class_id < len(source_names):
                        raise ValueError(f'Invalid class in {label_name}')
                    coordinates = values[1:]
                    if not all(-1e-6 <= v <= 1.000001 for v in coordinates):
                        raise ValueError(f'Out-of-bounds annotation in {label_name}')
                    coordinates = [min(1, max(0, v)) for v in coordinates]
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
                        if task == 'cavity':
                            omitted['invalid_cavity_box'] += 1
                            continue
                        raise ValueError(f'Empty box in {label_name}')
                    source_label = str(config['names'][int(class_id)])
                    target_id = 0
                    if task == 'fdi':
                        if source_label not in FDI_CLASSES:
                            omitted[source_label] += 1
                            continue
                        target_id = FDI_CLASSES.index(source_label)
                    elif task == 'cavity':
                        target_id = int(class_id)
                    elif tooth_types:
                        if (len(source_label) != 2 or source_label[0] not in '1234'
                                or source_label[1] not in '12345678'):
                            continue
                        target_id = int(source_label[1]) - 1
                    boxes.append(f'{target_id} ' + ' '.join(f'{v:.8f}' for v in (x, y, width, height)))
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
            if prefix + name in names:
                (stage / name).write_bytes(source.read(prefix + name))
        dataset = {'path': str(destination), 'train': 'train/images',
                   'val': 'valid/images', 'test': 'test/images', 'names': dict(enumerate(output_classes))}
        (stage / 'data.yaml').write_text(yaml.safe_dump(dataset, sort_keys=False))
        report = {'archive_sha256': hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest(),
                  'source': config.get('roboflow', {}), 'task': task or ('types' if tooth_types else 'tooth'),
                  'omitted_boxes': dict(omitted), 'splits': counts,
                  'source_class_instances': dict(class_counts),
                  'annotation_shapes': dict(polygon_shapes),
                  'output_classes': output_classes,
                  'label_policy': ('Group valid permanent FDI labels by their second digit. '
                                   'Omit 327 and 56. No quadrant identification.'
                                   if tooth_types else {'fdi': '32 permanent FDI positions; other codes omitted.',
                                       'cavity': 'Source cavity and normal boxes converted from pixel coordinates.'}.get(
                                           task, 'All source tooth classes collapsed to tooth.')),
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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--tooth-types', action='store_true')
    modes.add_argument('--task', choices=['fdi', 'cavity'])
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.destination, args.tooth_types, args.task), indent=2))
