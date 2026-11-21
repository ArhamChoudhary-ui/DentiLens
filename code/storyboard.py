"""Build source-linked examination chapters without inferring procedures."""
import csv
from io import StringIO
from statistics import median

import cv2
import numpy as np


def collect_events(report, reviews, instruments, transcript):
    rows = []
    for finding in report['findings']:
        rows.append(dict(id=finding['id'], kind='tooth', start=finding['time'], end=finding['time'],
                         label=finding['label'], status='suggested', image=finding['image']))
    for review in reviews:
        if review['decision'] == 'confirmed':
            rows.append(dict(id=review['id'], kind='bookmark', start=review['time'], end=review['time'],
                             label=review['label'] or 'Confirmed bookmark', note=review['note'], status='confirmed',
                             fdi=review.get('fdi_number')))
    for review in (instruments or {}).get('reviews', []):
        if review['decision'] == 'confirmed':
            rows.append(dict(id=review['detection_id'], kind='instrument', start=review['time'], end=review['time'],
                             label=review['label'] or 'Reviewed instrument', note=review['note'], status='confirmed',
                             fdi=review.get('fdi_number'), linked_review=review.get('tracked_from')))
    for segment in (transcript or {}).get('segments', []):
        if segment.get('status') != 'removed':
            rows.append(dict(id=segment['id'], kind='audio', start=segment['start'], end=segment['end'],
                             label=segment['text'], status=segment.get('status','suggested'),
                             note=segment.get('note',''), fdi=segment.get('fdi_number')))
    return sorted(rows,key=lambda row:row['start'])


def scene_boundaries(source, video, change_events=()):
    cap = cv2.VideoCapture(str(source))
    step = max(1, round(video['fps'] / 2))
    start = 0.0
    boundaries = [start]
    previous = None
    event_frames = {round(time * video['fps'] / step) * step for time in change_events}
    minimum_length = max(3, video['duration'] / 199)
    maximum_length = max(12, minimum_length)
    try:
        for frame in range(video['frames']):
            if not cap.grab():
                raise ValueError(f'Could not read source frame {frame}.')
            if frame % step:
                continue
            ok,image = cap.retrieve()
            if not ok:
                raise ValueError(f'Could not read source frame {frame}.')
            gray=cv2.cvtColor(cv2.resize(image,(96,54),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
            change=float(np.mean(cv2.absdiff(previous,gray))) if previous is not None else 0.0
            time=frame/video['fps']
            if (frame in event_frames or change>.20*255 or time-start>=maximum_length) and time-start>=minimum_length and video['duration']-time>=2:
                boundaries.append(round(time,3))
                start=time
            previous=gray
    finally:
        cap.release()
    return boundaries+[video['duration']]


def generate(source, report, events):
    breakpoints = [row['start'] for row in events if row['kind'] in ('bookmark', 'instrument', 'audio')]
    bounds=scene_boundaries(source,report['video'],breakpoints)
    chapters=[]
    for index,(start,end) in enumerate(zip(bounds,bounds[1:])):
        linked=[row for row in events if row['start']<end and row['end']>=start]
        confirmed=next((row for row in linked if row['kind'] in {'bookmark','instrument'}),None)
        finding=next((row for row in linked if row['kind']=='tooth'),None)
        thumbnail=(confirmed or finding or {}).get('start',median((start,end)))
        chapters.append(dict(id=f'chapter-{index}',title=confirmed['label'] if confirmed else
                             'Tooth visible · model suggestion' if finding else 'Examination footage',
                             start=start,end=end,thumbnail=thumbnail,note='',reviewed=False))
    return chapters


def with_events(chapters, events):
    return [{**item,'events':[row for row in events if row['start']<item['end'] and row['end']>=item['start']]}
            for item in chapters]


def csv_summary(chapters):
    stream=StringIO()
    writer=csv.writer(stream)
    writer.writerow(['id','start_sec','end_sec','title','note','reviewed'])
    for row in chapters:
        title,note=(("'"+str(row[key])) if str(row[key]).lstrip().startswith(('=','+','-','@')) else row[key] for key in ('title','note'))
        writer.writerow([row['id'],row['start'],row['end'],title,note,row['reviewed']])
    return stream.getvalue()
