"""Exercise the full renderer and local examination routes with controlled video."""
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import time
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
        full = client.get(url + '/media/annotated.mp4', headers={'Range': 'bytes=0-63'})
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
                                label='Reviewed upper central incisor', note='Demo note')).json()
        assert confirmed['time'] == finding['time']
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
