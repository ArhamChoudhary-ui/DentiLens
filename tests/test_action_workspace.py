from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.action_workspace import ActionWorkspace


def test_action_settings_and_upload_rejection(tmp_path):
    workspace = ActionWorkspace(tmp_path)
    app = FastAPI()
    app.include_router(workspace.router)

    with TestClient(app) as client:
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
    workspace.pool.shutdown()
