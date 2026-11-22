"""Train real linear action heads on frozen VideoMAE embeddings.

CSV columns: video,start,end,verb,noun,label,split. Split is train or val.
Paths are relative to the CSV. Validation must use separate source videos.
"""
import argparse
import csv
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import torch
from egosight.model import Encoder
from egosight.schema import atomic_json
from egosight.video import VideoReader
from egosight.contract import StudentKit


def read_rows(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    required = {'video','start','end','verb','noun','label','split'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('CSV must have video,start,end,verb,noun,label,split columns')
    for row in rows:
        row['video'] = str((path.parent/row['video']).resolve())
        row['start'], row['end'] = float(row['start']), float(row['end'])
        if row['split'] not in {'train','val'} or any(not row[k].strip() for k in ('verb','noun','label')):
            raise ValueError('Missing class or unsupported split')
    groups = {split: {r['video'] for r in rows if r['split']==split} for split in ('train','val')}
    if not all(groups.values()) or groups['train'] & groups['val']:
        raise ValueError('Need nonempty train and val sets with no source-video overlap')
    # Also reject copied media with another filename across splits.
    def hash_video(video):
        with Path(video).open('rb') as f:
            return hashlib.file_digest(f,'sha256').hexdigest()
    hashes = {split: {hash_video(v) for v in paths} for split,paths in groups.items()}
    if hashes['train'] & hashes['val']:
        raise ValueError('Identical video content in train and validation')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--allow-download', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    torch.manual_seed(2026)
    rows = read_rows(args.annotations.resolve())
    train = [r for r in rows if r['split']=='train']
    classes = {name: sorted({r[name] for r in train}) for name in ('verb','noun','label')}
    kit = StudentKit()
    kit.validate_classes(classes)
    if any(Path(row['video']).stem in kit.entries for row in train):
        raise ValueError('Training includes an organizer evaluation video. Use a separate permitted training split.')
    if any(r[k] not in classes[k] for r in rows for k in classes):
        raise ValueError('Validation contains a class absent from training')
    encoder = Encoder(device=args.device, offline=not args.allow_download)
    features = []
    for i,row in enumerate(rows):
        reader = VideoReader(row['video'])
        frames, _ = reader.sample(row['start'],row['end'],encoder.frames)
        features.append(encoder.encode(frames))
        print(f'Embedded {i+1}/{len(rows)}', flush=True)
    x = torch.stack(features)
    tr = torch.tensor([r['split']=='train' for r in rows])
    heads, metrics = {}, {}
    predicted = {}
    for name,vocab in classes.items():
        y = torch.tensor([vocab.index(r[name]) for r in rows])
        head = torch.nn.Linear(encoder.dimension,len(vocab))
        optimizer = torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
        for _ in range(args.epochs):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(x[tr]),y[tr])
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            prediction = head(x[~tr]).argmax(dim=1)
            metrics[name+'_window_top1'] = float((prediction == y[~tr]).float().mean())
            predicted[name] = prediction == y[~tr]
        heads[name] = head.state_dict()
    metrics['action_pair_window_top1'] = float((predicted['verb'] & predicted['noun']).float().mean())
    metrics['validation_windows'] = int((~tr).sum())
    metrics['note'] = 'Held-out window classification only; NOT action mAP, coverage, highlight F1 or FPVSum video accuracy.'
    artifact = dict(format='dentilens-linear-v1', encoder=encoder.source, revision=encoder.revision,
                    classes=classes, pairs=sorted({(r['verb'],r['noun']) for r in train}),
                    heads=heads, validation=metrics,
                    annotations_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(artifact,args.output)
    atomic_json(args.output.with_suffix('.metrics.json'),metrics)
    print(metrics)


if __name__ == '__main__':
    main()
