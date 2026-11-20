"""Audit an instrument OBB export and build isolated splits without augmentation copies."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from hashlib import file_digest, sha256
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import yaml


def cross_split_perceptual_pairs(directory, splits):
    """Return dHash candidates for visual inspection, not verified duplicates."""
    hashes = {}
    for split in splits:
        values = []
        for file in (directory / split / 'images').iterdir():
            gray = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            values.append(int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), 'big'))
        hashes[split] = values
    return {f'{a}/{b}': sum((x ^ y).bit_count() <= 2 for x in hashes[a] for y in hashes[b])
            for a, b in combinations(splits, 2)}


def prepare(source, destination):
    destination = Path(destination).resolve()
    splits = {'train': 'train', 'valid': 'val', 'test': 'test'}
    groups, exact, audit = defaultdict(list), defaultdict(set), Counter()
    originals = Counter()
    selected_hashes = defaultdict(set)
    with ZipFile(source) as archive:
        configurations = [archive.read(item) for item in archive.infolist() if item.filename == 'data.yaml']
        if not configurations or len(set(configurations)) != 1:
            raise ValueError('The archive has missing or inconsistent dataset configs.')
        names = yaml.safe_load(configurations[0])['names']
        attribution = archive.read('README.dataset.txt').decode()
        if 'CC BY 4.0' not in attribution:
            raise ValueError('Confirm the dataset license before preparing these images.')
        files = set(archive.namelist())
        dimensions = Counter()
        for path in sorted(files):
            parts = Path(path).parts
            if len(parts) != 3 or parts[0] not in splits or parts[1] != 'images':
                continue
            split, _, name = parts
            if Path(name).suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            originals[split] += 1
            label_path = f'{split}/labels/{Path(name).stem}.txt'
            if label_path not in files:
                audit['missing_label'] += 1
                continue
            image = archive.read(path)
            decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_GRAYSCALE)
            if decoded is None:
                audit['invalid_image'] += 1
                continue
            dimensions[f'{decoded.shape[1]} × {decoded.shape[0]}'] += 1
            labels = archive.read(label_path).decode().splitlines()
            counts = Counter()
            for label in labels:
                if not label.strip():
                    continue
                fields = label.split()
                try:
                    code = int(fields[0])
                    points = np.asarray([float(v) for v in fields[1:]], dtype=np.float32).reshape(4, 2)
                    if code not in names or not np.isfinite(points).all() or points.min() < 0 or points.max() > 1:
                        raise ValueError
                    if abs(cv2.contourArea(points)) < 1e-6 or not cv2.isContourConvex(points):
                        raise ValueError
                except ValueError:
                    audit['invalid_polygon'] += 1
                    continue
                counts[names[code]] += 1
            if sum(counts.values()) != sum(bool(line.strip()) for line in labels):
                audit['excluded_label_file'] += 1
                continue
            digest = sha256(image).hexdigest()
            group = name.split('.rf.')[0].lower()
            groups[group].append((split, path, label_path, digest, counts))
            exact[digest].add(split)
        selected, counts = Counter(), defaultdict(Counter)
        for variants in groups.values():
            if len({entry[0] for entry in variants}) != 1:
                audit['cross_split_source_groups'] += 1
                continue
            unique = next((entry for entry in variants if len(exact[entry[3]]) == 1), None)
            if not unique:
                audit['source_groups_excluded_for_exact_leakage'] += 1
                continue
            split, img, label, digest, classes = unique
            if digest in selected_hashes[split]:
                audit['same_split_exact_duplicates'] += 1
                continue
            selected_hashes[split].add(digest)
            for kind, member in (('images', img), ('labels', label)):
                folder = destination / split / kind
                folder.mkdir(parents=True, exist_ok=True)
                (folder / Path(member).name).write_bytes(archive.read(member))
            selected[split] += 1
            counts[split].update(classes)
        config = {key: f'{split}/images' for split, key in splits.items()}
        config.update(path=str(destination), names=names)
        (destination / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        (destination / 'ATTRIBUTION.txt').write_text(attribution)
    with Path(source).open('rb') as stream:
        digest = file_digest(stream, 'sha256').hexdigest()
    return dict(archive_sha256=digest, original_images=originals, prepared_images=selected,
                sizes=dimensions, checks=audit, classes=names, prepared_class_counts=counts,
                perceptual_candidates=cross_split_perceptual_pairs(destination, tuple(splits)),
                policy='One variant per source name, cross-split source names and exact image matches excluded.',
                limitations=['Patient IDs unavailable.', 'Visually similar images with different source names may remain.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('destination', type=Path)
    options = parser.parse_args()
    if options.destination.exists():
        parser.error('Use a new preparation folder.')
    result = prepare(options.archive, options.destination)
    output = json.dumps(result, indent=2)
    (options.destination / 'audit.json').write_text(output + '\n')
    print(output)
