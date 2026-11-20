"""Build a video review reel with annotated holds on exact source frames."""
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import tempfile
import unicodedata

import cv2
import imageio_ffmpeg
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_HEIGHT = 138
MAX_LABELS = 4


def context_windows(times, duration):
    """Merge overlapping three-second clips around selected review moments."""
    windows = []
    for time in sorted(times):
        start = min(max(0, time - 1), max(0, duration - 3))
        end = min(start + 3, duration)
        if windows and start <= windows[-1]['end_sec']:
            windows[-1]['end_sec'] = max(windows[-1]['end_sec'], end)
        else:
            windows.append(dict(start_sec=start, end_sec=end))
    return windows


def review_marks(report, reviews, keys=None):
    """Join original model predictions with reviewer decisions and manual bookmarks."""
    markers = {f'model-{i}': row for i, row in enumerate(report['observations'])}
    if keys is None:
        keys = markers
    marks = []
    for key in keys:
        review = reviews.get(key, {})
        if review.get('decision') == 'Dismiss':
            continue
        observation = markers.get(key)
        if observation is not None:
            marks.append({**observation, 'review': review})
        elif key.startswith('manual-') and review:
            timestamp = review['timestamp_sec']
            index = min(report['frames_decoded'] - 1, round(timestamp * report['fps']))
            marks.append(dict(source_frame=index, timestamp_sec=index / report['fps'], review=review))
    return marks


def clipped_text(text, width, scale):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'replace').decode().strip()
    text = ' '.join(text.split())
    if cv2.getTextSize(text, FONT, scale, 1)[0][0] <= width:
        return text
    while text and cv2.getTextSize(text + '...', FONT, scale, 1)[0][0] > width:
        text = text[:-1]
    return text.rstrip() + '...'


def mark_text(mark):
    review = mark['review']
    if review.get('tooth'):
        label = review['tooth']
    elif 'fdi_number' in mark:
        label = f"FDI {mark['fdi_number']} - {mark['tooth_name']}"
    else:
        label = mark.get('model_label', 'Manual bookmark')
    status = 'Reviewer confirmed' if review.get('decision') == 'Confirm' else 'AI suggestion'
    extras = []
    if mark.get('nearby_tooth_suggestion'):
        extras.append('Overlaps suggested ' + mark['nearby_tooth_suggestion'])
    if mark.get('identity_status') == 'conflicting':
        extras.append(f"FDI {mark['fdi_candidate']} or {mark['type_candidate']}")
    if review.get('flag'):
        extras.append('Flagged for review')
    if review.get('note'):
        extras.append(review['note'])
    return ' | '.join([status, label, *extras])


def mark_color(mark):
    if mark['review'].get('flag') or mark.get('category') == 'cavity_candidate':
        return (75, 190, 255)
    if 'mirror' in mark.get('model_label', ''):
        return (246, 190, 76)
    return (98, 230, 143)


def draw_region(frame, mark, number, evidence, used):
    bounds = mark.get('bbox_xyxy_normalized')
    if not bounds:
        return
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [round(value * size) for value, size in zip(bounds, [width, height] * 2)]
    color = mark_color(mark)
    mask = None
    if mark.get('mask_path'):
        mask = cv2.imread(str(evidence / mark['mask_path']), cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours((resized > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 2)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Give nearby teeth separate number markers while the panel carries full labels.
    origin_x = min(max(x1, 2), width - 23)
    origin_y = min(max(y1 - 20, 1), height - 22)
    while any(abs(origin_x - x) < 23 and abs(origin_y - y) < 20 for x, y in used):
        origin_y -= 21
        if origin_y < 1:
            return
    used.append((origin_x, origin_y))
    cv2.line(frame, (origin_x + 9, origin_y + 17), (x1, y1), color, 1)
    cv2.rectangle(frame, (origin_x, origin_y), (origin_x + 19, origin_y + 18), color, -1)
    cv2.putText(frame, str(number), (origin_x + 5, origin_y + 14), FONT, .45, (18, 24, 25), 1,
                cv2.LINE_AA)


def render_frame(frame, timestamp, marks, evidence):
    frame = frame.copy()
    height, width = frame.shape[:2]
    canvas = cv2.copyMakeBorder(frame, 0, PANEL_HEIGHT, 0, 0, cv2.BORDER_CONSTANT,
                               value=(26, 26, 24))
    heading = f'Source {timestamp:.2f} s  |  ' + ('Paused for review' if marks else 'Examination footage')
    cv2.putText(canvas, heading, (16, height + 27), FONT, .62, (245, 246, 241), 1, cv2.LINE_AA)
    used = []
    for index, mark in enumerate(marks[:MAX_LABELS], 1):
        draw_region(canvas[:height], mark, index, evidence, used)
        text = clipped_text(f'{index}. {mark_text(mark)}', width - 32, .49)
        cv2.putText(canvas, text, (16, height + 30 + 19 * index), FONT, .49,
                    mark_color(mark), 1, cv2.LINE_AA)
    if len(marks) > MAX_LABELS:
        cv2.putText(canvas, f'+{len(marks) - MAX_LABELS} suggestions in the report',
                    (16, height + 126), FONT, .45, (209, 210, 205), 1, cv2.LINE_AA)
    return canvas


def export_annotated_reel(source, destination, report, windows, marks, evidence):
    """Preserve moving source frames and hold annotated frames for one second."""
    source, destination, evidence = map(Path, (source, destination, evidence))
    if report.get('input_sha256'):
        with source.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != report['input_sha256']:
                raise ValueError('The source video differs from the video used for this analysis.')
    capture = cv2.VideoCapture(str(source))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (not capture.isOpened() or not math.isclose(fps, report['fps'], abs_tol=1e-3)
                or count != report['frames_decoded'] or min(width, height) < 2):
            raise ValueError('The source video timing could not be verified.')
        scale = min(1, 960 / max(width, height))
        output_width = max(240, 2 * round(width * scale / 2))
        output_height = max(2, 2 * round(height * output_width / width / 2))
        ranges = [(max(0, math.ceil(clip['start_sec'] * fps - 1e-7)),
                   min(count, math.ceil(clip['end_sec'] * fps - 1e-7))) for clip in windows]
        marks_at_frame = defaultdict(list)
        for mark in marks:
            marks_at_frame[mark['source_frame']].append(mark)
        if not any(first <= index < end for first, end in ranges for index in marks_at_frame):
            raise ValueError('No active predictions fall inside these windows. Choose a different moment.')
        inserted = set()
        expected_frames = 0
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.review-reel-') as tmp:
            staged = Path(tmp) / 'reel.mp4'
            writer = imageio_ffmpeg.write_frames(
                str(staged), (output_width, output_height + PANEL_HEIGHT),
                pix_fmt_in='bgr24', fps=fps, codec='libx264', macro_block_size=1,
                ffmpeg_timeout=20, output_params=['-movflags', '+faststart', '-preset', 'fast'])
            try:
                writer.send(None)
                for first, end in ranges:
                    if first >= end:
                        continue
                    if not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
                        raise ValueError('Could not seek to the selected review window.')
                    for index in range(first, end):
                        ok, frame = capture.read()
                        if not ok or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != index + 1:
                            raise ValueError('Could not read the expected source frame.')
                        frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
                        annotations = marks_at_frame.get(index, ())
                        canvas = render_frame(frame, index / fps, annotations, evidence)
                        repeats = max(1, round(fps)) if annotations else 1
                        expected_frames += repeats
                        if annotations:
                            inserted.add(index)
                        for _ in range(repeats):
                            writer.send(canvas)
            finally:
                writer.close()
            if not inserted:
                raise ValueError('No selected observations are inside the export windows.')
            check = cv2.VideoCapture(str(staged))
            try:
                if not check.isOpened() or round(check.get(cv2.CAP_PROP_FRAME_COUNT)) != expected_frames:
                    raise ValueError('The exported reel did not pass its frame-count check.')
            finally:
                check.release()
            staged.replace(destination)
        return {'annotated_frames': len(inserted), 'duration_sec': expected_frames / fps}
    finally:
        capture.release()
