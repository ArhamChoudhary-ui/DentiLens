"""Build evidence-linked views of an examination from its saved frame records."""
from collections import defaultdict
import json
import re

import cv2
import numpy as np

FDI = re.compile(r'(?<!\d)[1-4][1-8](?!\d)')
TYPE = re.compile(r'(?:(?:central|lateral) incisor|(?:first|second|third) (?:premolar|molar)|incisor|canine|premolar|molar)s?\b')


def appearances(rows):
    """Split observations into consecutive frame ranges without filling gaps."""
    spans = []
    for row in rows:
        if spans and row['frame'] == spans[-1]['last_frame'] + 1:
            spans[-1].update(last_frame=row['frame'], end=row['time'], count=spans[-1]['count'] + 1)
        else:
            spans.append(dict(frame=row['frame'], last_frame=row['frame'], time=row['time'], end=row['time'], count=1))
    return spans


def index_exam(timeline, report):
    tracks = defaultdict(list)
    findings = {item['id']: item for item in report['findings']}
    evidence = defaultdict(list)
    for item in findings.values():
        evidence[item.get('track_id')].append(item)
    with timeline.open() as stream:
        for line in stream:
            frame = json.loads(line)
            for i, row in enumerate(frame['annotations']):
                if row['category'] != 'tooth_identity' or not row.get('track_id'):
                    continue
                record = dict(row, frame=frame['frame'], time=frame['time'])
                record.setdefault('detection_id', f"{frame['frame']}-{i}")
                tracks[row['track_id']].append(record)
    return tracks, findings, evidence


def tooth_history(timeline, report, reviews, fdi=None, track_id=None):
    tracks, findings, evidence = index_exam(timeline, report)
    linked_reviews = defaultdict(list)
    for review in reviews:
        finding = findings.get(review.get('finding_id'), {})
        if finding.get('track_id'):
            linked_reviews[finding['track_id']].append(review)
    histories = []
    for key, all_rows in tracks.items():
        if track_id and key != track_id:
            continue
        approved = [item for item in linked_reviews[key]
                    if item['decision'] == 'confirmed' and item.get('fdi_number')]
        confirmed_codes = sorted({item['fdi_number'] for item in approved})
        rows = [row for row in all_rows if fdi is None or row.get('fdi_number') == fdi]
        if not rows and fdi not in confirmed_codes:
            continue
        observed_frames = rows or []
        histories.append(dict(track_id=key, suggested_numbers=sorted({row['fdi_number'] for row in all_rows if row.get('fdi_number')}),
                              confirmed_numbers=confirmed_codes, frames=observed_frames,
                              views=appearances(observed_frames), evidence=evidence[key],
                              visible_seconds=round(len(observed_frames) / report['video']['fps'], 3),
                              reviews=linked_reviews[key]))
    manual = [item for item in reviews if fdi and item.get('fdi_number') == fdi and
              not findings.get(item.get('finding_id'), {}).get('track_id')]
    return {'fdi': fdi, 'tracks': histories, 'other_reviews': manual}


def search_exam(timeline, report, reviews, query):
    tracks, findings, evidence = index_exam(timeline, report)
    q = query.lower()
    timestamp = re.search(r'\b(\d{1,2}):(\d{2})(\.\d+)?\b', q)
    if timestamp and int(timestamp[2]) >= 60:
        return {'results': [], 'message': 'Use a timestamp such as 00:09.50.'}
    exact = (int(timestamp[1]) * 60 + int(timestamp[2]) + float(timestamp[3] or 0)) if timestamp else None
    number = FDI.search(q if exact is None else q[:timestamp.start()] + q[timestamp.end():])
    kind = TYPE.search(q)
    code = number[0] if number else None
    tooth_type = kind[0].removesuffix('s') if kind else None
    if any(word in q for word in ('mirror', 'instrument', 'procedure', 'transcript', 'audio')):
        return {'results': [], 'message': 'This examination has no indexed instrument, activity, or audio records.'}
    if 'visibility' in q or 'blur' in q or 'exposure' in q:
        return {'results': [], 'message': 'Open Recording quality to find measured camera and lighting cues.'}

    def matches(item):
        return ((not code or item.get('fdi_number') == code) and
                (not tooth_type or tooth_type in (item.get('tooth_type') or item['label']).lower()) and
                (exact is None or item['frame'] == round(exact * report['video']['fps'])))

    results = []
    reviews_only = 'note' in q or 'bookmark' in q or ('confirmed' in q and 'unconfirmed' not in q)
    if 'clearest' in q or 'best view' in q:
        candidates = [item for item in findings.values() if matches(item)
                      and item['category'] == 'tooth_identity' and item.get('clarity') is not None]
        if candidates:
            best = max(candidates, key=lambda item: item['clarity'])
            return {'results': [dict(kind='track', id=best['id'], label=best['label'], time=best['time'],
                                     image=best['image'], confidence=best['confidence'], status='suggested')],
                    'total': 1, 'message': 'Highest clarity score among matching saved evidence frames.'}
        return {'results': [], 'message': 'No matching evidence with a recorded clarity score.'}
    if reviews_only or code and 'unconfirmed' not in q:
        for item in reviews:
            if item['decision'] != 'confirmed' or ('note' in q and not item['note']):
                continue
            finding = findings.get(item.get('finding_id'), {})
            if matches({**finding, **item, 'frame': round(item['time'] * report['video']['fps'])}):
                results.append(dict(kind='bookmark', id=item['id'], label=item['label'] or 'Confirmed bookmark',
                                    time=item['time'], image=finding.get('image'), note=item['note'], status='confirmed'))
    if not reviews_only and (code or tooth_type or timestamp or 'unconfirmed' in q):
        reviewed = {item.get('finding_id') for item in reviews}
        for key, rows in tracks.items():
            images = {item['frame']: item for item in evidence[key]}
            hits = {item['frame']: item for item in rows if matches(item) and
                    ('unconfirmed' not in q or images.get(item['frame'], {}).get('id') not in reviewed)}
            for span in appearances(hits.values()):
                frame = span['frame']
                row = hits[frame]
                finding = images.get(frame, {})
                results.append(dict(kind='track', id=row['detection_id'], track_id=key, label=row['label'],
                                    time=row['time'], end=span['end'], confidence=row.get('confidence'),
                                    image=finding.get('image'), status='suggested'))
    elif not reviews_only:
        return {'results': [], 'message': 'Try a tooth name, FDI number, timestamp, or “show notes”.'}
    results.sort(key=lambda item: item['time'])
    return {'results': results[:100], 'total': len(results),
            'message': '' if results else 'No matching records in this examination.'}


def evidence_index(timeline, report, reviews):
    tracks, findings, _ = index_exam(timeline, report)
    return {
        'schema_version': 1,
        'video': report['video'],
        'tracks': [{'id': key, 'observations': rows} for key, rows in tracks.items()],
        'evidence': list(findings.values()),
        'reviews': reviews,
    }


def quality_report(source, video):
    """Measure recording conditions. These cues describe images, not tooth visibility."""
    cap = cv2.VideoCapture(str(source))
    observations = defaultdict(list)
    previous = None
    try:
        for index in range(video['frames']):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Could not read frame {index}.')
            gray = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            cues = []
            if np.mean(gray <= 32) > .65:
                cues.append('Dark frame')
            if np.mean(gray >= 245) > .32:
                cues.append('Bright frame')
            if cv2.Laplacian(gray, cv2.CV_32F).var() < 22:
                cues.append('Low detail')
            if previous is not None and cv2.absdiff(previous, gray).mean() > 22:
                cues.append('Large frame change')
            for cue in cues:
                observations[cue].append(dict(frame=index, time=index / video['fps']))
            previous = gray
    finally:
        cap.release()
    return {'frames': video['frames'], 'cues': [
        {'label': label, 'frames': len(rows), 'appearances': appearances(rows)}
        for label, rows in sorted(observations.items())]}
