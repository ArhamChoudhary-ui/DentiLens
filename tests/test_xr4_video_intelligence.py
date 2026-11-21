import time

from fastapi.testclient import TestClient

from test_full_exam import examination_video, processor
from backend.main import create_app


def test_chapters_transcript_and_instrument_anchors(monkeypatch, examination_video, tmp_path):
    import backend.main as backend

    def recognize(source, models, language, duration):
        return dict(state='ready', model='test-decoder', seconds=.1, language=language,
                    segments=[dict(id='audio-0', start=.25, end=1.5, text='Review the upper front tooth.')])

    monkeypatch.setattr(backend, 'transcribe', recognize)
    with TestClient(create_app(tmp_path/'exams', processor, serve_frontend=False)) as client:
        response = client.post('/api/exams', files={'video': ('fixture.mp4', examination_video.read_bytes(), 'video/mp4')})
        assert response.status_code == 202
        path = '/api/exams/' + response.json()['id']
        for _ in range(120):
            if client.get(path).json()['state'] == 'ready':
                break
            time.sleep(.05)

        transcript = client.post(path + '/transcript', json={'language': 'en'}).json()
        original = transcript['original_segments'][0]
        assert original['start'] == .25
        result = client.put(path + '/transcript/audio-0', json={
            **original, 'end': 1.4, 'text': 'Review the front tooth.', 'fdi_number': '21'})
        assert result.json()['original_segments'] == [original]
        assert result.json()['segments'][0]['text'] == 'Review the front tooth.'
        assert client.put(path + '/transcript/audio-0', json={**original, 'end': 8}).status_code == 422
        assert client.post(path + '/voice-note', files={'recording': ('voice.webm', b'fixture')}).json()['text'] == original['text']

        response = client.post(path + '/instrument-reviews', json={
            'time': .8, 'decision': 'confirmed', 'label': 'Reviewed instrument', 'box': [.1,.1,.75,.8]})
        assert response.status_code == 200
        detection = response.json()['detection_id']
        track = client.post(path + '/instrument-reviews/' + detection + '/track')
        assert track.status_code == 200 and track.json()['anchor'] == detection
        assert all(row['frame'] != 8 for row in track.json()['frames'])
        assert client.post(path + '/instrument-reviews/instrument-0-1/track').status_code == 404
        assert len(client.get(path + '/instrument-annotations').json()['annotations']) == 1

        story = client.post(path + '/storyboard').json()
        first = story['chapters'][0]
        assert first['start'] == 0 and story['chapters'][-1]['end'] == 3
        assert {'audio','tooth','instrument'} <= {event['kind'] for event in first['events']}
        edit = client.put(path + '/storyboard', json={'chapters': [{**first, 'title': '=Reviewed', 'reviewed': True}]})
        assert edit.status_code == 200, edit.text
        assert edit.json()['original_chapters'] == story['original_chapters']
        assert "'=Reviewed" in client.get(path + '/storyboard-export?format=csv').text
        assert client.get(path + '/storyboard-export').json()['chapters'][0]['reviewed']
        assert client.get(path + '/intelligence', params={'q':'show transcript'}).json()['results'][0]['time'] == .25
        assert client.put(path + '/storyboard', json={'chapters': [first, first]}).status_code == 422
