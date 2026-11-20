"""Render a new full recording from saved observations and a snapshot of reviews."""
from collections import defaultdict
import json
from pathlib import Path
import tempfile

import cv2
import imageio_ffmpeg

from dental_modes import fdi_name
from full_video import ffmpeg, stamp, video_info


def review_index(report, reviews):
    findings = {item['id']: item for item in report['findings']}
    tracks, frames = {}, {}
    for review in reviews:
        finding = findings.get(review.get('finding_id'))
        frame = finding['frame'] if finding else min(report['video']['frames'] - 1, round(review['time'] * report['video']['fps']))
        if finding and finding['category'] == 'tooth_identity' and finding.get('track_id'):
            tracks[finding['track_id']] = (review, finding)
        elif finding or review.get('box'):
            frames.setdefault(frame, []).append((review, finding))
    return tracks, frames


def corrected_rows(entry, tracks, frames, width, height):
    output = []
    local = frames.get(entry['frame'], [])
    handled = set()
    for row in entry['annotations']:
        pair = tracks.get(row.get('track_id'))
        if pair is None:
            pair = next((item for item in local if item[1] and
                         item[1]['category'] == row['category'] and item[1]['box'] == row['box']), None)
        if pair is None or pair[0]['decision'] != 'confirmed':
            continue
        review, finding = pair
        handled.add(review['id'])
        output.append(confirmed_row(row, review, finding, entry['frame'], width, height))
    for review, finding in local:
        if review['id'] not in handled and review['decision'] == 'confirmed' and review.get('box'):
            base = {'category': 'tooth_identity', 'label': 'Marked region', 'confidence': 0}
            output.append(confirmed_row(base, review, finding, entry['frame'], width, height))
    return output


def confirmed_row(row, review, finding, frame, width, height):
    item = dict(row)
    item.pop('fdi_number', None)
    item.pop('tooth_type', None)
    box = review.get('box')
    if box and (not finding or frame == finding['frame']):
        item['box'] = [value * extent for value, extent in zip(box, [width, height] * 2)]
    code = review.get('fdi_number')
    if code:
        item['fdi_number'] = code
    item['label'] = review['label'] or (fdi_name(code) if code else 'Confirmed region')
    item['review_status'] = 'confirmed'
    return item


def instrument_marks(record):
    reviews = {item['detection_id']: item for item in record.get('reviews', [])}
    frames = defaultdict(list)
    for row in record.get('observations', []):
        review = reviews.get(row['detection_id'], {})
        if review.get('decision') != 'confirmed':
            continue
        geometry = dict(box=review['box'], polygon=None) if review.get('box') else {}
        frames[row['frame']].append({**row, **geometry, 'label': review.get('label') or 'Reviewed instrument',
                                     'category': 'instrument', 'review_status': 'confirmed'})
    return frames


def render_reviewed(work, report, reviews, options, instruments=None):
    video = report['video']
    width, height, fps = (video[key] for key in ('width', 'height', 'fps'))
    reviewed = options['version'] == 'confirmed'
    tracks, frames = review_index(report, reviews)
    instrument_frames = instrument_marks(instruments or {})
    source = work / 'source.mp4'
    with tempfile.TemporaryDirectory(dir=work, prefix='.rerender-') as temporary:
        raw = Path(temporary) / 'video.mp4'
        output = Path(temporary) / 'export.mp4'
        capture = cv2.VideoCapture(str(source))
        writer = imageio_ffmpeg.write_frames(str(raw), (width, height), pix_fmt_in='bgr24',
            fps=fps, codec='libx264', macro_block_size=1, output_params=['-preset', 'veryfast'])
        count = 0
        try:
            writer.send(None)
            with (work / 'timeline.jsonl').open() as timeline:
                for line in timeline:
                    entry = json.loads(line)
                    ok, frame = capture.read()
                    if not ok or entry['frame'] != count:
                        raise ValueError('The saved frame records do not match the recording.')
                    rows = (corrected_rows(entry, tracks, frames, width, height) if reviewed else entry['annotations'])
                    extra = []
                    for row in instrument_frames.get(count, []):
                        box = [value * extent for value, extent in zip(row['box'], [width, height] * 2)]
                        extra.append({**row, 'box': box})
                    writer.send(stamp(frame, entry['time'], rows + extra, options=options))
                    count += 1
        finally:
            writer.close()
            capture.release()
        ffmpeg('-i', raw, '-i', source, '-map', '0:v:0', '-map', '1:a:0?', '-c', 'copy',
               '-t', f"{video['duration']:.8f}", '-movflags', '+faststart', output)
        if count != video['frames'] or video_info(output)['frames'] != count:
            raise ValueError('The export did not preserve the recording frame count.')
        destination = work / f"{options['version']}-export.mp4"
        output.replace(destination)
    return destination
