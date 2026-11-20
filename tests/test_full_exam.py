"""Exercise the full renderer and local examination routes with controlled video."""
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import time
from threading import Event
from zipfile import ZipFile

from fastapi.testclient import TestClient
import imageio_ffmpeg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.main import create_app
from full_video import ffmpeg, process_exam, video_info


@pytest.fixture
def examination_video(tmp_path):
    target = tmp_path / 'examination.mp4'
    ffmpeg('-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=10:duration=2',
           '-f', 'lavfi', '-i', 'color=c=white:size=320x180:rate=10:duration=1',
           '-f', 'lavfi', '-i', 'sine=frequency=440:duration=3',
           '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[v]',
           '-map', '[v]', '-map', '2:a', '-c:v', 'libx264', '-c:a', 'aac', target)
    return target


def processor(source, destination, profile, device, cavities, progress):
    def detect(frame):
        if frame.mean() > 240:
            return []
        return [dict(category='tooth_identity', bbox_xyxy_normalized=[.05, .12, .85, .77],
                     confidence=.84, tooth_type='central incisor', fdi_number='11',
                     identity_status='type agreement')]

    return process_exam(source, destination, profile, device, cavities, progress, detect)


def test_examination_workflow(examination_video, tmp_path):
    storage = tmp_path / 'sessions'
    with TestClient(create_app(storage, processor, serve_frontend=False)) as client:
        wrong = client.post('/api/exams', files={'video': ('test.txt', b'bad', 'text/plain')})
        assert wrong.status_code == 415
        with examination_video.open('rb') as stream:
            upload = client.post('/api/exams', data={'profile': 'detailed', 'device': 'cpu'},
                                 files={'video': ('examination.mp4', stream, 'video/mp4')})
        assert upload.status_code == 202
        exam = upload.json()
        assert exam['metadata']['fps'] == 10
        assert exam['metadata']['duration'] == 3
        url = '/api/exams/' + exam['id']
        assert client.get(url + '/preview', headers={'Range': 'bytes=0-31'}).status_code == 206

        for _ in range(200):
            exam = client.get(url).json()
            if exam['state'] in ('ready', 'failed'):
                break
            time.sleep(.05)
        assert exam['state'] == 'ready', exam.get('error')
        report = exam['report']
        assert report['video']['frames'] == report['stats']['frames_decoded'] == 30
        assert 0 < report['stats']['detection_frames'] < 30
        assert report['stats']['tracked_frames'] > 0
        assert report['stats']['numbered_frames'] > 0

        timeline = client.get(url + '/media/timeline.jsonl').text.splitlines()
        frames = [json.loads(line) for line in timeline]
        assert [entry['frame'] for entry in frames] == list(range(30))
        assert all(not entry['annotations'] for entry in frames[20:])
        assert sum(report['frame_reasons'].values()) == 30
        assert all('detection_id' in row for frame in frames for row in frame['annotations'])
        numbered = client.get(url + '/tooth-history', params={'fdi': '11'}).json()
        assert len(numbered['tracks']) == 1
        track = numbered['tracks'][0]
        assert all(row['fdi_number'] == '11' for row in track['frames'])
        assert track['visible_seconds'] == len(track['frames']) / 10
        lookup = client.get(url + '/tooth-history', params={'track': track['track_id']}).json()
        assert len(lookup['tracks'][0]['frames']) > len(track['frames'])
        ask = client.get(url + '/intelligence', params={'q': 'Find tooth 11'}).json()
        assert ask['results'][0]['label'].startswith('FDI 11')
        assert ask['results'][0]['time'] == track['frames'][0]['time']
        assert not client.get(url + '/intelligence', params={'q': 'Mirror'}).json()['results']
        quality = client.get(url + '/quality').json()
        assert quality['frames'] == 30
        bright = next(cue for cue in quality['cues'] if cue['label'] == 'Bright frame')
        assert bright['appearances'] == [{'frame': 20, 'last_frame': 29, 'time': 2., 'end': 2.9, 'count': 10}]
        index = client.get(url + '/evidence-index').json()
        assert index['tracks'][0]['id'] == track['track_id']
        media_url = url + '/media/annotated.mp4'
        headers = client.head(media_url)
        assert headers.status_code == 200 and headers.headers['content-type'] == 'video/mp4'
        full = client.get(media_url, headers={'Range': 'bytes=0-63'})
        assert full.status_code == 206 and len(full.content) == 64
        output = storage / exam['id'] / 'annotated.mp4'
        assert video_info(output)['frames'] == 30
        streams = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-i', str(output)],
                                 capture_output=True, text=True).stderr
        assert 'Video: h264' in streams and 'Audio: aac' in streams

        finding = next(item for item in report['findings'] if item['label'].startswith('FDI 11'))
        assert client.get(url + '/images/' + finding['image']).headers['content-type'] == 'image/jpeg'
        images = client.get(url + '/evidence.zip')
        with ZipFile(BytesIO(images.content)) as archive:
            assert archive.namelist() == [item['image'] for item in report['findings']]
        assert client.post(url + '/reviews', json={'time': 3, 'label': 'Outside recording'}).status_code == 422
        confirmed = client.post(url + '/reviews', json=dict(time=0, finding_id=finding['id'],
                                label='Reviewed upper central incisor', note='Demo note', fdi_number='21',
                                box=[.1,.2,.7,.6])).json()
        assert confirmed['time'] == finding['time']
        assert confirmed['original_ai']['label'] == finding['label']
        lookup = client.get(url + '/tooth-history', params={'fdi':'21'}).json()
        assert lookup['tracks'][0]['frames'] == []
        assert lookup['tracks'][0]['confirmed_numbers'] == ['21']
        notes = client.get(url + '/intelligence', params={'q':'Show notes for tooth 21'}).json()
        assert notes['results'][0]['note'] == 'Demo note'
        assert notes['results'][0]['status'] == 'confirmed'
        assert client.post(url + '/reviews', json={'time': 0, 'fdi_number': '99'}).status_code == 422
        assert client.get(url + '/still.jpg', params={'bookmark': confirmed['id']}).headers['content-type'] == 'image/jpeg'
        reviewed = client.post(url + '/video-export', json={'version': 'confirmed'})
        assert reviewed.status_code == 200, reviewed.text
        assert client.get(reviewed.json()['url'], headers={'Range':'bytes=0-63'}).status_code == 206
        corrected = storage / exam['id'] / 'confirmed-export.mp4'
        assert video_info(corrected) == video_info(output)
        reviewed_streams = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-i', str(corrected)],
                                          capture_output=True, text=True).stderr
        assert 'Audio: aac' in reviewed_streams
        assert client.get(url + '/export').json()['review_history'][0]['fdi_number'] == '21'
        manual = client.post(url + '/reviews', json=dict(time=2.2, label='XR bookmark')).json()
        reel = client.post(url + '/reel', json={'bookmarks': [confirmed['id'], manual['id']]})
        assert reel.status_code == 200, reel.text
        assert reel.json()['annotated_frames'] == 2
        assert client.get(reel.json()['url']).headers['content-type'] == 'video/mp4'
        assert client.get(url).json()['reel_url'] == reel.json()['url']
        saved = client.get(url + '/export').json()['reviewer_notes']
        assert [(item['label'], item['decision']) for item in saved] == [
            ('Reviewed upper central incisor', 'confirmed'), ('XR bookmark', 'confirmed')]
        client.post(url + '/reviews', json=dict(time=0, finding_id=finding['id'], decision='dismissed'))
        assert client.post(url + '/reel', json={'bookmarks': [confirmed['id']]}).status_code == 422

    with TestClient(create_app(storage, processor, serve_frontend=False)) as restarted:
        assert restarted.get(url).json()['state'] == 'ready'
        assert restarted.get(url).json()['reviews'][0]['label'] == 'XR bookmark'


def test_cancelled_analysis_can_be_retried(examination_video, tmp_path):
    started, release = Event(), Event()

    def blocked(source, destination, profile, device, cavities, progress):
        started.set()
        release.wait(4)
        progress(.25, 'Checking cancellation')
        return processor(source, destination, profile, device, cavities, progress)

    with TestClient(create_app(tmp_path / 'cancel', blocked, serve_frontend=False)) as client:
        with examination_video.open('rb') as stream:
            item = client.post('/api/exams', data={'device': 'cpu'},
                               files={'video': ('examination.mp4', stream, 'video/mp4')}).json()
        assert started.wait(4)
        url = '/api/exams/' + item['id']
        try:
            assert client.post(url + '/cancel').json()['state'] == 'canceling'
        finally:
            release.set()
        for _ in range(100):
            if client.get(url).json()['state'] == 'cancelled':
                break
            time.sleep(.05)
        assert client.get(url).json()['state'] == 'cancelled'
        assert client.post(url + '/retry').status_code == 202
        for _ in range(100):
            if client.get(url).json()['state'] == 'ready':
                break
            time.sleep(.05)
        assert client.get(url).json()['state'] == 'ready'
