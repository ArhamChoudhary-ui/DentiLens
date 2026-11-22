"""Train real linear action heads on frozen VideoMAE embeddings.

CSV columns: video,start,end,verb,noun,label,split. Split is train or val.
Paths are relative to the CSV. Validation must use separate source videos.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
import math
from time import perf_counter
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
        if not math.isfinite(row['end']) or not 0 <= row['start'] < row['end']:
            raise ValueError('Invalid training interval')
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
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--encoder-path')
    parser.add_argument('--allow-download', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.patience) < 1:
        parser.error('epochs and patience must be positive')
    if args.output.exists() or args.output.with_suffix('.metrics.json').exists():
        parser.error('Choose a new output path to preserve previous training artifacts')
    started = perf_counter()
    torch.manual_seed(2026)
    rows = read_rows(args.annotations.resolve())
    train = [r for r in rows if r['split']=='train']
    classes = {name: sorted({r[name] for r in train}) for name in ('verb','noun','label')}
    kit = StudentKit()
    kit.validate_classes(classes)
    if any(Path(row['video']).stem in kit.entries for row in rows):
        raise ValueError('The training or validation split includes an organizer evaluation video.')
    if any(r[k] not in classes[k] for r in rows for k in classes):
        raise ValueError('Validation contains a class absent from training')
    encoder_options = {'source': args.encoder_path} if args.encoder_path else {}
    encoder = Encoder(device=args.device, offline=not args.allow_download, **encoder_options)
    features = []
    for i,row in enumerate(rows):
        reader = VideoReader(row['video'])
        if row['end'] > reader.info.duration:
            raise ValueError(f"Annotation exceeds source duration: {row['video']}")
        frames, _ = reader.sample(row['start'],row['end'],encoder.frames)
        features.append(encoder.encode(frames))
        print(f'Embedded {i+1}/{len(rows)}', flush=True)
    x = torch.stack(features)
    tr = torch.tensor([r['split']=='train' for r in rows])
    heads, metrics = {}, {}
    predicted, training_log = {}, {}
    for name,vocab in classes.items():
        y = torch.tensor([vocab.index(r[name]) for r in rows])
        head = torch.nn.Linear(encoder.dimension,len(vocab))
        optimizer = torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
        best_loss, best_epoch, best_state = math.inf, 0, None
        history = training_log[name] = []
        for epoch in range(1, args.epochs + 1):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(x[tr]),y[tr])
            loss.backward()
            optimizer.step()
            with torch.inference_mode():
                validation_loss = float(torch.nn.functional.cross_entropy(head(x[~tr]), y[~tr]))
            if not math.isfinite(validation_loss) or not math.isfinite(float(loss.detach())):
                raise ValueError(f'Nonfinite {name} loss')
            history.append({'epoch': epoch, 'train_loss': float(loss.detach()), 'validation_loss': validation_loss})
            if validation_loss < best_loss:
                best_loss, best_epoch, best_state = validation_loss, epoch, deepcopy(head.state_dict())
            if epoch - best_epoch >= args.patience:
                break
        head.load_state_dict(best_state)
        metrics[name+'_best_epoch'] = best_epoch
        with torch.inference_mode():
            prediction = head(x[~tr]).argmax(dim=1)
            metrics[name+'_window_top1'] = float((prediction == y[~tr]).float().mean())
            predicted[name] = prediction == y[~tr]
        heads[name] = head.state_dict()
    metrics['action_pair_window_top1'] = float((predicted['verb'] & predicted['noun']).float().mean())
    metrics['training_windows'] = len(train)
    metrics['elapsed_seconds'] = perf_counter() - started
    metrics['configuration'] = {'seed': 2026, 'epochs': args.epochs, 'patience': args.patience,
                                'encoder': encoder.source, 'encoder_revision': encoder.revision, 'device': encoder.device}
    metrics['learning_curves'] = training_log
    metrics['validation_windows'] = int((~tr).sum())
    metrics['note'] = 'Held-out window classification only; NOT action mAP, coverage, highlight F1 or FPVSum video accuracy.'
    artifact = dict(format='egosight-linear-v1', encoder=encoder.source, revision=encoder.revision,
                    classes=classes, pairs=sorted({(r['verb'],r['noun']) for r in train}),
                    heads=heads, validation=metrics,
                    annotations_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    torch.save(artifact, temporary)
    temporary.replace(args.output)
    atomic_json(args.output.with_suffix('.metrics.json'),metrics)
    print(metrics)


if __name__ == '__main__':
    main()
