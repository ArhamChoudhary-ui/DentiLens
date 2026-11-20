"""Complete examination rendering with periodic AI detections and guarded tracking."""
from collections import Counter
import json
import math
import shutil
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

import cv2
import imageio_ffmpeg
import numpy as np

from exam_tracking import ExamTracker, bridge

PROFILES = {'fast': 2.0, 'balanced': 1.0, 'detailed': .5}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def video_info(path):
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = (int(capture.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0 or min(count, width, height) <= 0:
            raise ValueError('The video could not be decoded. Try an MP4 or MOV with a standard video codec.')
        return dict(fps=fps, frames=count, width=width, height=height, duration=count / fps)
    finally:
        capture.release()


def ffmpeg(*args, timeout=600):
    try:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                        '-nostdin', '-y', *map(str, args)], check=True, capture_output=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        raise ValueError('Video encoding failed: ' + exc.stderr.decode(errors='replace').strip()[-550:]) from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError('Video encoding timed out. Try a shorter recording.') from exc


def normalize_video(source, target, metadata):
    fps = min(metadata['fps'], 30)
    # Scale down for local review while keeping the full recording and its aspect ratio.
    vf = f"scale=w='min(iw,1280)':h='min(ih,1280)':force_original_aspect_ratio=decrease:force_divisible_by=2,fps={fps:.8g}"
    ffmpeg('-i', source, '-map', '0:v:0', '-map', '0:a:0?', '-vf', vf, '-c:v', 'libx264',
           '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-movflags', '+faststart', target)


def safe_text(text, limit=76):
    return text.replace('·', '|').encode('ascii', 'replace').decode()[:limit]


def stamp(frame, timestamp, rows, options=None):
    """Use short numbered markers and a label rail so neighboring text stays readable."""
    height, width = frame.shape[:2]
    options = options or {}
    show_text = options.get('labels', True) or options.get('fdi', False) or options.get('scores', False) or options.get('track_ids', False)
    show_time = options.get('timestamp', True)
    legend = 'Confirmed bookmarks' if options.get('version') == 'confirmed' else 'AI suggestions'
    ui_scale = min(1.6, max(.55, width / 750))
    columns = 2 if width >= 360 else 1
    header_height = round(23 * ui_scale)
    line_height = round(22 * ui_scale)
    padding = round(8 * ui_scale)
    rows_fit = min(4, max(0, (height // 3 - header_height) // line_height))
    chosen = sorted(rows, key=lambda row: (row['category'] != 'cavity_candidate',
                                            row['category'] == 'tooth_region', -row.get('confidence', 0)))[:columns * rows_fit]
    panel_height = min(height, header_height + line_height * math.ceil(len(chosen) / columns) * show_text)
    top = height - panel_height
    cv2.rectangle(frame, (0, top), (width, height), (23, 25, 25), -1)
    stroke = 2 if ui_scale >= 1.2 else 1
    cv2.putText(frame, f'{legend} | {timestamp:.2f}s' if show_time else legend, (padding, top + header_height - padding),
                FONT, .44 * ui_scale, (239, 244, 242), stroke, cv2.LINE_AA)
    placed = []
    marker_size = round(16 * ui_scale)
    for index, row in enumerate(chosen, 1):
        cavity = row['category'] == 'cavity_candidate'
        color = (85, 191, 255) if cavity else (126, 223, 178)
        x1, y1, x2, y2 = map(round, row['box'])
        if options.get('boxes', True):
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, stroke + 1)
        if not show_text:
            continue
        anchor_x = max(0, min(x1, width - marker_size))
        y = max(1, min(y1 - marker_size - 1, top - marker_size - 1))
        for previous_x, previous_y in placed:
            if abs(previous_x - anchor_x) < marker_size and abs(previous_y - y) < marker_size:
                y = max(1, previous_y - marker_size - 1)
        placed.append((anchor_x, y))
        cv2.rectangle(frame, (anchor_x, y), (anchor_x + marker_size, y + marker_size), color, -1)
        number = str(index)
        number_scale = .43 * ui_scale
        (number_width, number_height), _ = cv2.getTextSize(number, FONT, number_scale, 1)
        cv2.putText(frame, number, (anchor_x + (marker_size - number_width) // 2,
                                    y + (marker_size + number_height) // 2),
                    FONT, number_scale, (22, 26, 26), 1, cv2.LINE_AA)
        line, col = divmod(index - 1, columns)
        label = row['label']
        if options and not options.get('fdi', True) and label.startswith('FDI ') and ' · ' in label:
            label = label.split(' · ', 1)[1]
        parts = [label] if options.get('labels', True) else []
        if options.get('fdi', False) and row.get('fdi_number') and not (parts and label.startswith('FDI ')):
            parts.insert(0, f"FDI {row['fdi_number']}")
        if options.get('scores', False) and row.get('review_status') != 'confirmed' and row.get('confidence') is not None:
            parts.append(f"{row['confidence']:.0%}")
        if options.get('track_ids', False) and row.get('track_id'):
            parts.append(row['track_id'])
        text = safe_text(f"{index}. " + ' · '.join(parts))
        label_scale = .47 * ui_scale
        text_width = cv2.getTextSize(text, FONT, label_scale, stroke)[0][0]
        label_scale *= min(1, (width / columns - 2 * padding) / max(1, text_width))
        cv2.putText(frame, text, (padding + col * width // columns,
                                 top + header_height + (line + 1) * line_height - padding),
                    FONT, label_scale, color, stroke, cv2.LINE_AA)
    return frame


def local_predictor(models, device, cavities):
    from combined_review import CombinedReview

    predictor = CombinedReview(models, models / 'grounded', device=device)

    def detect(frame):
        diagnostics = {}
        rows, _ = predictor.predict(frame, include_region=True, include_cavities=cavities, diagnostics=diagnostics)
        return rows, diagnostics

    return detect


def process_exam(source, destination, profile, device, cavities, progress, detect=None):
    started = perf_counter()
    destination, source = Path(destination), Path(source)
    original = video_info(source)
    if original['duration'] > 600:
        raise ValueError('This local prototype supports recordings up to 10 minutes.')
    progress(.03, 'Preparing browser playback')
    with tempfile.TemporaryDirectory(dir=destination, prefix='.full-') as temporary:
        stage = Path(temporary)
        normalized = stage / 'source.mp4'
        normalize_video(source, normalized, original)
        media = video_info(normalized)
        fps, frame_count = media['fps'], media['frames']
        width, height = media['width'], media['height']
        if abs(media['duration'] - original['duration']) > max(.25, 2 / fps):
            # OpenCV may only estimate the duration of a variable-frame-rate file.
            media['source_timing_note'] = 'The source was converted to constant frame rate for tracking.'
        progress(.12, 'Loading tooth models')
        model_started = perf_counter()
        detect = detect or local_predictor(Path(__file__).resolve().parents[1] / 'models', device, cavities)
        model_seconds = perf_counter() - model_started
        inference_seconds = 0
        reasons = Counter()
        tracker = ExamTracker(fps)
        cadence = max(1, round(PROFILES[profile] * fps))
        refresh_gap = max(1, round(.33 * fps))
        cap = cv2.VideoCapture(str(normalized))
        writer = imageio_ffmpeg.write_frames(str(stage / 'video-only.mp4'), (width, height),
            pix_fmt_in='bgr24', fps=fps, codec='libx264', macro_block_size=1,
            output_params=['-preset', 'veryfast'], ffmpeg_timeout=20)
        tracks, favorites = {}, {}
        stats = Counter(frames_decoded=0, detection_frames=0, annotated_frames=0, tracked_frames=0,
                        numbered_frames=0, cavity_frames=0)
        previous = None
        last_detection = -cadence
        needs_refresh = False
        try:
            writer.send(None)
            with (stage / 'timeline.jsonl').open('w') as timeline:
                for index in range(frame_count):
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f'Video ended at frame {index} of {frame_count}.')
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cut = previous is not None and float(cv2.absdiff(previous[::8, ::8], gray[::8, ::8]).mean()) > 35
                    if cut:
                        tracker.reset()
                    propagated = bridge(previous, gray, tracker.active) if previous is not None and not cut else []
                    lost = len(propagated) < len(tracker.active)
                    needs_refresh = needs_refresh or lost
                    refresh = (index - last_detection >= cadence or cut
                               or (needs_refresh and index - last_detection >= refresh_gap))
                    flags = []
                    diagnostics = {}
                    if refresh:
                        input_rows = []
                        inference_started = perf_counter()
                        predicted = detect(frame)
                        inference_seconds += perf_counter() - inference_started
                        predictions, diagnostics = predicted if isinstance(predicted, tuple) else (predicted, {})
                        for row in predictions:
                            bounds = row['bbox_xyxy_normalized']
                            if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
                                continue
                            coordinates = [float(min(1, max(0, value)) * extent) for value, extent in
                                           zip(bounds, [width, height] * 2)]
                            if coordinates[2] - coordinates[0] < 2 or coordinates[3] - coordinates[1] < 2:
                                continue
                            if row['category'] == 'cavity_candidate':
                                flags.append({**row, 'label': 'Possible cavity', 'box': coordinates, 'origin': 'detected'})
                            else:
                                input_rows.append({**row, 'box': coordinates})
                        last_detection = index
                        needs_refresh = False
                        stats['detection_frames'] += 1
                    else:
                        input_rows = propagated
                    rows = tracker.update(input_rows, (width, height), index, refresh) + flags
                    now = index / fps
                    stats['frames_decoded'] += 1
                    stats['annotated_frames'] += bool(rows)
                    stats['tracked_frames'] += any(r['origin'] == 'tracked' for r in rows)
                    stats['numbered_frames'] += any(r.get('fdi_number') for r in rows)
                    stats['individual_tooth_frames'] += any(r['category'] == 'tooth_identity' for r in rows)
                    stats['tracking_loss_frames'] += lost
                    if rows:
                        reason = 'overlay'
                    elif refresh and diagnostics.get('region_score', 1) < .38:
                        reason = 'region_below_search_threshold'
                    elif refresh and not input_rows:
                        reason = 'no_accepted_detection'
                    elif refresh:
                        reason = 'association_rejected'
                    elif needs_refresh:
                        reason = 'tracking_lost'
                    else:
                        reason = 'between_detection_passes'
                    reasons[reason] += 1
                    stats['cavity_frames'] += bool(flags)
                    for row in rows:
                        key = row.get('track_id')
                        if key:
                            track = tracks.setdefault(key, dict(id=key, start=now, end=now,
                                                                 label=row['label'], frames=0, anchors=0))
                            track.update(end=now, frames=track['frames'] + 1)
                            track['anchors'] += row['origin'] == 'detected'
                            if row.get('fdi_number') or (not track.get('fdi_number') and row.get('tooth_type')):
                                track.update(label=row['label'], fdi_number=row.get('fdi_number'),
                                             tooth_type=row.get('tooth_type'))
                        if row['origin'] == 'detected':
                            favorite = key or f'cavity-{int(now / 3)}'
                            x1, y1, x2, y2 = map(round, row['box'])
                            crop = gray[y1:y2, x1:x2]
                            sharpness = cv2.Laplacian(cv2.resize(crop, (48, 48)), cv2.CV_32F).var()
                            light = 1 - min(1, abs(float(crop.mean()) - 135) / 135)
                            clarity = round(.8 * min(1, math.log1p(float(sharpness)) / 8) + .2 * light, 3)
                            rank = (2 if row.get('fdi_number') else 1 if row.get('tooth_type') else 0) + row['confidence'] + .5 * clarity
                            if rank > favorites.get(favorite, {}).get('rank', -1):
                                favorites[favorite] = dict(rank=rank, frame=index, time=now,
                                    track_id=key, label=row['label'], category=row['category'],
                                    confidence=row['confidence'], box=row['box'], clarity=clarity,
                                    reason='Representative detection', tooth_type=row.get('tooth_type'),
                                    fdi_number=row.get('fdi_number'))
                    compact = [{**{k: row[k] for k in ('label', 'box', 'category', 'origin', 'track_id', 'fdi_number',
                                                       'confidence', 'tooth_type', 'type_score', 'number_score', 'flow_score')
                                   if k in row}, 'detection_id': f'{index}-{i}', 'review_status': 'suggested'}
                               for i, row in enumerate(rows)]
                    timeline.write(json.dumps(dict(frame=index, time=now, annotations=compact, reason=reason,
                                                   diagnostics=diagnostics), separators=(',', ':'))+'\n')
                    writer.send(stamp(frame, now, rows))
                    previous = gray
                    if index % 15 == 0 or index + 1 == frame_count:
                        progress(.15 + .75 * (index + 1) / frame_count, f'Annotated frame {index + 1:,} of {frame_count:,}')
            if cap.read()[0]:
                raise ValueError('The video contains more frames than its normalized metadata reports.')
        finally:
            cap.release()
            writer.close()

        progress(.92, 'Finishing video and review images')
        output = stage / 'annotated.mp4'
        ffmpeg('-i', stage / 'video-only.mp4', '-i', normalized, '-map', '0:v:0', '-map', '1:a:0?',
               '-t', f'{media["duration"]:.8f}', '-c', 'copy', '-movflags', '+faststart', output)
        encoded = video_info(output)
        if encoded['frames'] != frame_count or abs(encoded['duration'] - media['duration']) > 1 / fps:
            raise ValueError('The full-video export did not match the normalized source timeline.')
        image_dir = stage / 'images'
        image_dir.mkdir()
        photos = sorted(sorted(favorites.values(), key=lambda item: -item['rank'])[:20], key=lambda item: item['time'])
        cap = cv2.VideoCapture(str(normalized))
        try:
            for i, item in enumerate(photos):
                cap.set(cv2.CAP_PROP_POS_FRAMES, item['frame'])
                ok, frame = cap.read()
                if not ok:
                    continue
                item.pop('rank')
                path = f'finding-{i:02d}.jpg'
                item['image'] = path
                item['id'] = f'ai-{i:02d}'
                if not cv2.imwrite(str(image_dir / path), stamp(frame, item['time'], [{**item, 'origin': 'detected'}])):
                    raise ValueError('Could not save an examination evidence image.')
        finally:
            cap.release()
        report = dict(input=original, video=media, device=device, profile=profile, cavity_review=cavities,
                      stats=dict(stats), frame_reasons=dict(reasons), eligible_frames=None,
                      performance=dict(model_load_seconds=round(model_seconds, 2),
                                       average_detection_seconds=round(inference_seconds / max(1, stats['detection_frames']), 3),
                                       total_seconds=round(perf_counter() - started, 2)),
                      tracks=sorted(tracks.values(), key=lambda item: item['start']),
                      findings=[item for item in photos if 'id' in item],
                      limitations=['Labels are AI suggestions. FDI requires repeated consistent detections.',
                                   'Frames without reliable detections or tracks retain the full footage without boxes.',
                                   'Tracks are local to this recording. Camera cuts and occlusion can break continuity.',
                                   'Possible cavity flags are for research review.'],
                      tracking='ByteTrack association + forward/backward optical flow')
        (stage / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        progress(.98, 'Publishing recording')
        if (destination / 'images').exists():
            shutil.rmtree(destination / 'images')
        for filename in ('source.mp4', 'annotated.mp4', 'timeline.jsonl', 'images', 'report.json'):
            (stage / filename).replace(destination / filename)
        return report
