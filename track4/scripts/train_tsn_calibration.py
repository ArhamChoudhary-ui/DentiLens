"""Fit a lightweight residual calibration head on official EPIC training windows."""
import argparse
import csv
from hashlib import file_digest
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import cv2
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from egosight.contract import StudentKit
from egosight.pretrained import EpicTSN, frame_tensor


def seconds(value):
    hours, minutes, seconds = map(float, value.split(':'))
    return hours * 3600 + minutes * 60 + seconds


def extract(model, rows, path):
    capture = cv2.VideoCapture(str(path))
    images, samples = [], []
    try:
        for row in sorted(rows, key=lambda r: r['start_timestamp']):
            start, end = map(seconds, (row['start_timestamp'], row['stop_timestamp']))
            if not math.isfinite(end) or not 0 <= start < end:
                raise ValueError(f'Invalid training timestamps: {path.name}')
            frames = []
            for fraction in (.25, .75):
                capture.set(cv2.CAP_PROP_POS_MSEC, 1000 * (start + (end-start) * fraction))
                ok, frame = capture.read()
                if ok:
                    frames.append(frame_tensor(frame))
            if len(frames) == 2:
                images.extend(frames)
                samples.append((int(row['verb_class']), int(row['noun_class'])))
    finally:
        capture.release()
    if not samples:
        raise ValueError(f'No annotated windows could be decoded from {path.name}')
    scale = torch.tensor([.485, .456, .406])[None, :, None, None]
    with torch.inference_mode():
        logits = torch.cat([model((batch.float() / 255 - scale) / scale)
                            for batch in torch.stack(images).split(8)]).reshape(-1, 2, 397).mean(1)
    return logits.clone(), torch.tensor(samples)


def loss(logits, targets):
    return F.cross_entropy(logits[:, :97], targets[:, 0]) + F.cross_entropy(logits[:, 97:], targets[:, 1])


def evaluate(logits, targets):
    verbs = logits[:, :97].argmax(1) == targets[:, 0]
    nouns = logits[:, 97:].argmax(1) == targets[:, 1]
    return {'loss': float(loss(logits, targets)), 'verb_top1': float(verbs.float().mean()),
            'noun_top1': float(nouns.float().mean()), 'pair_top1': float((verbs & nouns).float().mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations', type=Path)
    parser.add_argument('videos', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--checksums', type=Path, required=True, help='Official EPIC downloader md5.csv')
    parser.add_argument('--train', nargs='+', required=True)
    parser.add_argument('--val', nargs='+', required=True)
    args = parser.parse_args()
    ids = args.train + args.val
    if len(set(ids)) != len(ids) or {v.split('_')[0] for v in args.train} & {v.split('_')[0] for v in args.val}:
        parser.error('Use distinct source participants between training and validation')
    if set(ids) & StudentKit().entries.keys() or args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error('Use non-benchmark source IDs and a new checkpoint path')
    with args.annotations.open(newline='') as file:
        rows = list(csv.DictReader(file))
    with args.checksums.open(newline='') as file:
        checksums = {r['file_remote_path']: r['md5'] for r in csv.DictReader(file)}
    for vid in ids:
        path = args.videos / (vid + '.mp4')
        participant = vid.split('_')[0]
        new_path = f"{participant}/videos/{vid}.MP4"
        old_path = f"videos/train/{participant}/{vid}.MP4"
        official = checksums.get(new_path, checksums.get(old_path))
        with path.open('rb') as video:
            if official != file_digest(video, 'md5').hexdigest():
                raise ValueError(f'Official EPIC video checksum mismatch: {vid}')
    with args.annotations.open('rb') as file:
        annotations_sha256 = file_digest(file, 'sha256').hexdigest()
    start = perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    pretrained = EpicTSN(ROOT / 'weights', 'cpu')
    manifest = json.loads((ROOT / 'weights/epic-classes.json').read_text())
    if annotations_sha256 != manifest['sources_sha256']['EPIC_100_train.csv']:
        raise ValueError('Annotations do not match the official EPIC training CSV')
    pretrained.model.requires_grad_(False)
    data = {}
    for vid in ids:
        selected = [r for r in rows if r['video_id'] == vid]
        if not selected or any(r['participant_id'] != vid.split('_')[0]
                               or not 0 <= int(r['verb_class']) < 97
                               or not 0 <= int(r['noun_class']) < 300 for r in selected):
            raise ValueError(f'Invalid official annotations for {vid}')
        data[vid] = extract(pretrained.model, selected, args.videos / (vid + '.mp4'))
        print(vid, len(data[vid][1]), flush=True)
    x, y = (torch.cat(items) for items in zip(*(data[v] for v in args.train)))
    vx, vy = (torch.cat(items) for items in zip(*(data[v] for v in args.val)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    interim = args.output.with_suffix('.best.tmp')
    bias = torch.nn.Parameter(torch.zeros(397))
    log_scale = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW([bias, log_scale], lr=.01)
    def predict(values):
        return values * log_scale.exp().repeat_interleave(torch.tensor([97, 300])) + bias
    history = []
    best = None
    for epoch in range(1, 41):
        optimizer.zero_grad()
        train_loss = loss(predict(x), y) + .1 * bias.square().mean()
        train_loss.backward()
        optimizer.step()
        with torch.no_grad():
            val = evaluate(predict(vx), vy)
        history.append({'epoch': epoch, 'train_loss': float(train_loss.detach()), **val})
        if best is None or val['loss'] < best[0]['loss']:
            best = (val, epoch, bias.detach().clone(), log_scale.detach().clone())
            torch.save({'format': 'egosight-tsn-calibration-v1', 'bias': best[2], 'log_scale': best[3],
                        'pretrained_sha256': pretrained.checksum, 'epoch': epoch}, interim)
        if epoch - best[1] >= 8:
            break
    val, epoch, bias, log_scale = best
    metrics = {'annotations_sha256': annotations_sha256, 'seed': 2026, 'device': 'cpu',
               'frames_per_window': 2, 'learning_rate': .01, 'max_epochs': 40, 'patience': 8,
               'pretrained_sha256': pretrained.checksum, 'train_sources': args.train, 'validation_sources': args.val,
               'train_windows': len(y), 'validation_windows': len(vy), 'baseline': evaluate(vx, vy),
               'validation': val, 'best_epoch': epoch, 'seconds': perf_counter() - start}
    torch.save({'format': 'egosight-tsn-calibration-v1', 'bias': bias, 'log_scale': log_scale,
                'pretrained_sha256': pretrained.checksum, 'metrics': metrics}, interim)
    interim.replace(args.output)
    args.output.with_suffix('.json').write_text(json.dumps({**metrics, 'history': history}, indent=2))
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
