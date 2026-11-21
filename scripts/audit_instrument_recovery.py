"""Replay the 4.1 proposals and the original photo splits with production inference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from zipfile import ZipFile

from audit_instrument_crops import ROOT, read_clip
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import torch
import yaml

from instrument_identity import (DEFAULT_CROP_PADDING, classify_pipeline, crop_bounds, describe_tracks,
                                 encoded_crop, identity_model, score_crops)
from train_instrument_identity import photo_data
from dentalguard_crops import dentalguard


def metrics(model, info, rows):
    classes = info['classes']
    truth = [label for _, label in rows]
    predictions = []
    for start in range(0, len(rows), 32):
        predictions.extend(score_crops(model, info, [image for image, _ in rows[start:start+32]],
                                       diagnostics=True))
    top = [prediction['class_id'] for prediction in predictions]
    confidence = np.array([p['confidence'] for p in predictions])
    correct = np.equal(truth, top)
    accepted = [p for p in predictions if p['accepted']]
    accepted_by_class = {}
    for label in classes:
        tp = sum(p['accepted'] and p['class_id']==label and actual==label
                 for actual,p in zip(truth,predictions))
        proposed = sum(p['class_id']==label for p in accepted)
        positives = truth.count(label)
        accepted_by_class[label] = dict(matches=tp, predictions=proposed, positives=positives,
                                        precision=tp/proposed if proposed else None,
                                        recall=tp/positives if positives else None,
                                        f1=2*tp/(proposed+positives) if proposed+positives else None)
    bins = np.minimum((confidence*10).astype(int), 9)
    calibration = sum(
        (mask.sum()/len(rows))*abs(correct[mask].mean()-confidence[mask].mean())
        for i in range(10) if (mask:=bins==i).any()
    )
    return dict(
        accepted_by_class=accepted_by_class, coverage=len(accepted)/len(rows),
        known_name_rate_on_unknown=sum(p['accepted'] for label,p in zip(truth,predictions) if label=='unknown')/truth.count('unknown') if 'unknown' in truth else None,
        raw_top1_expected_calibration_error_10_bins=float(calibration),
        crops=len(rows), classes=classes, truth_counts=dict(Counter(truth)), predicted_counts=dict(Counter(top)),
        top1=float(np.mean(np.equal(truth, top))),
        top3=float(np.mean([label in [item['class_id'] for item in p['alternatives']]
                            for label, p in zip(truth, predictions)])),
        macro_f1=float(f1_score(truth, top, labels=classes, average='macro', zero_division=0)),
        confusion_matrix=confusion_matrix(truth, top, labels=classes).tolist(),
        rejected=sum(not p['accepted'] for p in predictions),
        rejection_reasons=dict(Counter(p['reason'] for p in predictions if not p['accepted'])),
        top_confidence_p10_p50_p90=np.percentile([p['confidence'] for p in predictions], [10, 50, 90]).tolist(),
        margin_p10_p50_p90=np.percentile([p['margin'] for p in predictions], [10, 50, 90]).tolist(),
    )


def inspect_archive(path):
    with ZipFile(path) as archive:
        config = yaml.safe_load(archive.read('data.yaml'))
        readme = archive.read('README.roboflow.txt').decode()
        image_names = [name for name in archive.namelist() if "/images/" in name and name.endswith(".jpg")]
        image_hashes = {sha256(archive.read(name)).digest() for name in image_names}
        counts = Counter()
        groups = set()
        boxes = []
        bad_labels = 0
        for name in archive.namelist():
            if '/labels/' not in name or not name.endswith('.txt'):
                continue
            group = Path(name).stem.split('.rf.', 1)[0]
            if group in groups:
                continue
            groups.add(group)
            for line in archive.read(name).decode().splitlines():
                fields = line.split()
                if len(fields) != 9:
                    bad_labels += 1
                    continue
                counts[config['names'][int(fields[0])]] += 1
                corners = np.array(fields[1:], dtype=float).reshape(4, 2)
                boxes.append(np.ptp(corners.clip(0, 1), axis=0))
        return dict(yaml=config, export_notes=readme, source_filename_groups=len(groups),
                    image_files=len(image_names), exact_unique_images=len(image_hashes),
                    exact_duplicate_images=len(image_names)-len(image_hashes),
                    boxes_per_class_in_filename_groups=dict(counts), skipped_malformed_labels=bad_labels,
                    normalized_box_width_height_p10_p50_p90=np.percentile(boxes, [10, 50, 90], axis=0).tolist())


def replay(source, baseline, models, output):
    rows = [dict(item, confidence=item['detection_confidence']) for item in baseline['evidence']]
    raw = classify_pipeline(source, rows, models[:2], models[2])
    for row in rows:
        row['identity'] = raw[row['detection_id']]
    _, diagnostic = describe_tracks(rows, read_clip(source), [])
    canon = {row['detection_id']: row for row in rows}
    capture = cv2.VideoCapture(str(source))
    evidence = []
    tiles = []
    try:
        current_frame = -1
        for row in sorted(rows, key=lambda item: item['frame']):
            if row['frame'] != current_frame:
                for index in range(current_frame+1,row['frame']):
                    if not capture.grab():
                        raise ValueError(f'Could not skip to {source.name} frame {index}')
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Could not read {source.name} frame {row['frame']}")
                current_frame = row['frame']
            h, w = frame.shape[:2]
            chosen = canon[row.get('duplicate_of') or row['detection_id']]
            prefix = source.name.split('-', 1)[0].split('_', 1)[0] + '-' + row['detection_id']
            shown = frame.copy()
            x1, y1, x2, y2 = crop_bounds(row['box'], w, h, 0)
            cv2.rectangle(shown, (x1, y1), (x2, y2), (85, 255, 170), max(2, min(h, w)//170))
            image = prefix + '-frame.jpg'
            cv2.imwrite(str(output/image), shown)
            tile = np.full((195, 380, 3), 30, np.uint8)
            thumb = cv2.resize(shown, (round(w*145/max(h,w)),round(h*145/max(h,w))), interpolation=cv2.INTER_AREA)
            tile[24:24+thumb.shape[0], :thumb.shape[1]] = thumb
            cv2.putText(tile, f"{prefix.split('-')[0]} | {row['time']:.2f}s | {chosen['track_id']}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX,.46,(240,240,240),1,cv2.LINE_AA)
            heads = {}
            for index, (model, info) in enumerate(models):
                name = info['model']
                px = encoded_crop(frame, row['box'], info)
                bounds = crop_bounds(row['box'], w, h, info.get('crop_padding',DEFAULT_CROP_PADDING))
                crop_image = prefix + f'-crop-{index}.png'
                if px is not None:
                    bgr = cv2.cvtColor(px, cv2.COLOR_RGB2BGR) if px.ndim == 3 else cv2.cvtColor(px, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(str(output/crop_image), bgr)
                    tile[39:103, 150+index*75:214+index*75] = cv2.resize(bgr, (64,64))
                heads[name] = dict(prediction=score_crops(model, info, [px], diagnostics=True)[0],
                                   crop=crop_image if px is not None else None, pixel_bounds=bounds,
                                   width=bounds[2]-bounds[0], height=bounds[3]-bounds[1])
            cv2.putText(tile,row['reason_code'],(150,130),cv2.FONT_HERSHEY_SIMPLEX,.4,(232,232,232),1,cv2.LINE_AA)
            tiles.append(tile)
            evidence.append(dict(video=source.name, frame=row['frame'], time=row['time'], frame_image=image,
                                 detection_id=row['detection_id'], canonical_id=chosen['detection_id'],
                                 track_id=chosen['track_id'], box=row['box'], canonical_box=chosen['box'],
                                 detection_confidence=row['confidence'], status=row['instrument_status'],
                                 reason=row['reason_code'], heads=heads))
    finally:
        capture.release()
    return dict(source=source.name, diagnostics=diagnostic, evidence=evidence), tiles


def render(output, reports, tiles):
    tiles.extend([np.zeros_like(tiles[0])]*(-len(tiles)%3))
    cv2.imwrite(str(output/'contact-sheet.jpg'), np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))
    sections = []
    for report in reports:
        for row in report['evidence']:
            heads = []
            for model, data in row['heads'].items():
                prediction = data['prediction']
                ranks = ', '.join(f"{p['label']} {p['confidence']:.1%}" for p in (prediction or {}).get('alternatives',[]))
                crop_image = f'<img width="96" alt="Classifier crop" src="{data["crop"]}">' if data['crop'] else ''
                heads.append(f'<div>{crop_image}<b>{escape(model)}</b><br>{escape(ranks)}'
                             f'<br>{escape(str((prediction or {}).get("reason") or "Passed frame gate"))}</div>')
            sections.append(f'<article><h2>{escape(row["video"])} · {row["time"]:.2f} s · {row["track_id"]}</h2>'
                            f'<p>{escape(row["reason"])}</p><img width="480" alt="Frame with proposal box" '
                            f'src="{row["frame_image"]}"><div class="heads">{"".join(heads)}</div></article>')
    (output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Instrument audit</title>'
                                    '<style>body{font:16px system-ui;margin:2rem;color:#222}article{border-top:1px solid #aaa;'
                                    'padding:1rem 0}.heads{display:flex;gap:1rem;flex-wrap:wrap}.heads img{display:block}'
                                    'article>img{max-width:100%;max-height:360px;object-fit:contain}</style>'
                                    '<h1>Instrument recognition audit</h1><p><a href="contact-sheet.jpg">Contact sheet</a> · '
                                    '<a href="audit.json">Raw evidence and validation</a></p>'+''.join(sections), encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','archive','clips','output'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--dentalguard',type=Path)
    parser.add_argument('--mirror-probes',type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    stems = ['instrument-identity','instrument-detail','instrument-context']
    models = [identity_model(ROOT/'models',stem) for stem in stems]
    checkpoints = []
    validation = {}
    for index, (stem, (model, info)) in enumerate(zip(stems,models)):
        path = ROOT/'models'/f'{stem}.pt'
        checkpoints.append(dict(path=str(path.relative_to(ROOT)), sha256=info['weights_sha256'],
                                modified_at=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
                                **{key:info[key] for key in ('model','classes','thresholds','backbone','aliases')},
                                color=info.get('color',False)))
        if index < 2:
            splits = photo_data(args.archive, info['aliases'], info.get('color',False))
            validation[info['model']] = {
                'train_crop_counts_before_resampling':dict(Counter(label for _, label in splits['train'])),
                **{split:metrics(model,info,splits[split]) for split in ('valid','test')},
            }
    guard_validation = {}
    if args.dentalguard:
        for index, (model, info) in enumerate(models):
            splits, origins = dentalguard(args.dentalguard, context=index==2, color=info.get('color',False))
            if index==2 and origins != info['video_group_splits']:
                raise ValueError('DentalGuard groups do not match the context checkpoint.')
            guard_validation[info['model']] = {
                'source_groups':origins,
                **{split:metrics(model, info, splits[split]) for split in ('guard_valid','guard_test')},
            }
    reports, tiles = [], []
    for baseline in json.loads(args.baseline.read_text())['holdouts']:
        report, images = replay(args.clips/baseline['source'], baseline, models, args.output)
        reports.append(report)
        tiles.extend(images)
    probes = None
    if args.mirror_probes:
        manifest = json.loads(args.mirror_probes.read_text())
        source = args.clips/manifest['source']
        source_evidence = next(report['evidence'] for report in reports if report['source']==source.name)
        video = read_clip(source)
        reviewed = []
        for frame, box in manifest['frames'].items():
            index = int(frame)
            reviewed.append(dict(frame=index,time=index/video['fps'],detection_id=f'manual-mirror-{index}',
                                 box=(np.asarray(box)/np.tile(manifest['grid'],2)).tolist(),
                                 detection_confidence=None))
        probes, probe_tiles = replay(source, {'evidence':reviewed}, models, args.output)
        probes['label'] = manifest['label']
        proposals_by_frame = Counter(row['frame'] for row in source_evidence)
        probes['detector_proposals_on_reviewed_frames'] = {
            frame:proposals_by_frame[int(frame)] for frame in manifest['frames']
        }
        tiles.extend(probe_tiles)
    result = dict(checkpoints=checkpoints, dataset=inspect_archive(args.archive), photo_validation=validation, dentalguard_validation=guard_validation, videos=reports, manual_mirror_probes=probes)
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n', encoding='utf8')
    render(args.output,reports+([probes] if probes else []),tiles)


if __name__=='__main__':
    main()
