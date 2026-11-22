"""Serve the local EgoSight video review workspace."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.action_workspace import ActionWorkspace

ROOT = Path(__file__).resolve().parents[1]


def create_app(storage=None, serve_frontend=True):
    workspace = ActionWorkspace(Path(storage or ROOT / 'runtime/full-video/action-intelligence').resolve())

    @asynccontextmanager
    async def lifespan(_app):
        workspace.restore()
        try:
            yield
        finally:
            workspace.pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title='EgoSight XR', lifespan=lifespan)
    app.include_router(workspace.router)

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'models_ready': all(workspace.settings()['models'].values())}

    dist = ROOT / 'frontend/dist'
    if serve_frontend and (dist / 'index.html').is_file():
        app.mount('/', StaticFiles(directory=dist, html=True))
    return app


app = create_app()
