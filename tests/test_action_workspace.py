from fastapi.testclient import TestClient

from backend.main import create_app


def test_action_settings_and_upload_rejection(tmp_path):
    app = create_app(storage=tmp_path, serve_frontend=False)
    workspace = next(route.endpoint.__self__ for route in app.routes if route.path == "/api/action/settings")

    with TestClient(app) as client:
        assert client.get('/api/health').json()['status']=='ok'
        assert 'ski' in client.get('/api/action/settings').json()['labels']
        assert client.get('/api/action/runs').json() == []
        path = '/api/action/runs'
        assert client.post(path, files={'video': ('text.txt', b'bad')}).status_code == 415
        assert client.post(path, files={'video': ('unreadable.mp4', b'bad')}).status_code == 422
        assert not list(tmp_path.iterdir())
        assert workspace.slot.acquire(blocking=False)
        try:
            assert client.post(path, files={'video': ('busy.mp4', b'bad')}).status_code == 409
        finally:
            workspace.slot.release()
