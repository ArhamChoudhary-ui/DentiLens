"""Check instrument scan and review integration with deterministic test detections."""


from test_full_exam import examination_video, processor
from backend.main import create_app
from fastapi.testclient import TestClient
from instrument_video import group_instruments, measure_instruments
from full_video import video_info


def test_instrument_groups_do_not_fill_unsampled_gaps():
    def row(frame, label):
        return dict(frame=frame, time=frame / 25, detection_id=f'instrument-{frame}-0',
                    track_id='I-4', confidence=.8, **{'class': label})
    rows = [row(0, 'Dental mirror'), row(5, 'Dental mirror'), row(10, 'Forceps'), row(20, 'Dental mirror')]
    spans = group_instruments(rows, 5)
    assert [item['samples'] for item in spans] == [2, 1, 1]
    assert spans[0]['end'] == .2


def test_instrument_scan_api_and_full_video_export(monkeypatch, examination_video, tmp_path):
    import backend.main as backend
    import time
    polygon = [[.3,.2],[.65,.3],[.65,.65],[.3,.55]]
    def scan(path, video, models):
        return measure_instruments(path, video, models, detector=lambda frame: [('Dental mirror', .8, polygon, 7)])
    monkeypatch.setattr(backend, 'instrument_model_ready', lambda root: True)
    monkeypatch.setattr(backend, 'measure_instruments', scan)
    storage = tmp_path / 'exam-store'
    with TestClient(create_app(storage, processor, serve_frontend=False)) as client:
        response = client.post('/api/exams', files={'video':('fixture.mp4', examination_video.read_bytes(),'video/mp4')})
        exam = response.json()
        path = '/api/exams/' + exam['id']
        for _ in range(120):
            if client.get(path).json()['state'] == 'ready':
                break
            time.sleep(.05)
        measured = client.post(path + '/instruments')
        assert measured.status_code == 200, measured.text
        assert measured.json()['samples'] == 15
        assert len(measured.json()['observations']) == 15
        first = measured.json()['appearances'][0]['best']
        reviewed = client.post(path + '/instrument-reviews', json={
            'detection_id':first['detection_id'], 'decision':'confirmed', 'label':'Reviewed mirror', 'note':'Training fixture'})
        assert reviewed.json()['original_label'] == 'Dental mirror'
        indexed = client.get(path + '/intelligence', params={'q':'Show mirror'}).json()['results'][0]
        assert indexed['status'] == 'confirmed' and indexed['time'] == first['time']
        assert not client.get(path + '/intelligence', params={'q':'Instrument at 00:00.10'}).json()['results']
        at_time = client.get(path + '/intelligence', params={'q':'Instrument at 00:00.20'}).json()['results']
        assert at_time and all(item['time'] == .2 for item in at_time)
        index = client.get(path + '/evidence-index').json()
        assert index['instruments']['reviews'][0]['label'] == 'Reviewed mirror'
        result = client.post(path + '/video-export', json={'version':'confirmed','instruments':True})
        assert result.status_code == 200, result.text
        assert video_info(storage / exam['id'] / 'confirmed-export.mp4')['frames'] == 30
