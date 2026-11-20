"""Check instrument scan and review integration with deterministic test detections."""


from test_full_exam import examination_video, processor
from backend.main import create_app
from fastapi.testclient import TestClient
from instrument_video import group_instruments, measure_instruments
from full_video import video_info
from reviewed_video import instrument_marks


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
        manual = client.post(path + '/instrument-reviews', json={
            'time': .22, 'decision': 'confirmed', 'label': 'Wire cutter', 'box': [.03,.2,.4,.5]})
        assert manual.status_code == 200, manual.text
        assert manual.json()['original_label'] is None and manual.json()['time'] == .2
        measured = client.post(path + '/instruments')
        assert measured.status_code == 200, measured.text
        assert measured.json()['samples'] == 15
        assert len(measured.json()['observations']) == 16
        assert not client.get(path + '/intelligence', params={'q':'Show mirror'}).json()['results']
        marks = instrument_marks(measured.json())
        assert list(marks) == [2] and len(marks[2]) == 1
        first = measured.json()['appearances'][0]['best']
        reviewed = client.post(path + '/instrument-reviews', json={
            'detection_id':first['detection_id'], 'decision':'confirmed', 'label':'Reviewed explorer',
            'note':'Training fixture', 'box':[.05,.2,.4,.5]})
        assert reviewed.json()['original_label'] == 'Dental mirror'
        assert not client.get(path + '/intelligence', params={'q':'Show mirror'}).json()['results']
        indexed = client.get(path + '/intelligence', params={'q':'Show explorer'}).json()['results'][0]
        assert indexed['status'] == 'confirmed' and indexed['time'] == first['time']
        assert not client.get(path + '/intelligence', params={'q':'Instrument at 00:00.10'}).json()['results']
        at_time = client.get(path + '/intelligence', params={'q':'Instrument at 00:00.20'}).json()['results']
        assert len(at_time) == 1 and at_time[0]['label'] == 'Wire cutter'
        assert at_time[0]['time'] == .2
        index = client.get(path + '/evidence-index').json()
        assert len(index['instruments']['reviews']) == 2
        marks = instrument_marks(index['instruments'])
        assert marks[0][0]['box'] == [.05,.2,.4,.5] and marks[0][0]['polygon'] is None
        assert {'Wire cutter','Reviewed explorer'} == {row['label'] for entries in marks.values() for row in entries}
        assert 'observations' not in client.get(path + '/export').json()['instrument_scan']
        result = client.post(path + '/video-export', json={'version':'confirmed','instruments':True})
        assert result.status_code == 200, result.text
        assert video_info(storage / exam['id'] / 'confirmed-export.mp4')['frames'] == 30
