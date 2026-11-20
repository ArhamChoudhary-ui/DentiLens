"""Local FastAPI service for full-video examinations and reviewer notes."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from io import BytesIO
import json
import logging
from pathlib import Path
import shutil
import sqlite3
import sys
from typing import Annotated, Literal
from uuid import UUID, uuid4
from zipfile import ZipFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from annotated_reel import context_windows, export_annotated_reel
from full_video import process_exam, video_info

LOG = logging.getLogger('dentilens')
MAX_BYTES = 500 * 1024 * 1024
EXTENSIONS = {'.mp4', '.mov', '.m4v', '.webm'}
Profile = Literal['fast', 'balanced', 'detailed']
Device = Literal['auto', 'cpu', 'mps']
Asset = Literal['source.mp4', 'annotated.mp4', 'reviewed-reel.mp4', 'timeline.jsonl']


class Review(BaseModel):
    time: float = Field(ge=0, allow_inf_nan=False)
    finding_id: str | None = Field(default=None, pattern=r'^ai-\d{2}$')
    label: str = Field(default='', max_length=100)
    note: str = Field(default='', max_length=500)
    decision: Literal['confirmed', 'dismissed'] = 'confirmed'
    flagged: bool = False


class ReelRequest(BaseModel):
    bookmarks: list[str] = Field(min_length=1, max_length=20)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def models_ready():
    model_dir = ROOT / 'models'
    pairs = [f'{name}.{extension}' for name in ('tooth-fdi', 'tooth-types', 'cavity-review')
             for extension in ('pt', 'json')]
    return all((model_dir / name).is_file() for name in pairs + ['grounded/dino/model.safetensors'])


def create_app(storage=None, processor=process_exam, serve_frontend=True):
    storage = Path(storage or ROOT / 'runtime/full-video').resolve()
    database = storage / 'exams.sqlite3'
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dentilens-exam')

    def connect():
        database_connection = sqlite3.connect(database, timeout=15)
        database_connection.row_factory = sqlite3.Row
        return database_connection

    def exam_record(row):
        item = dict(row)
        item['metadata'] = json.loads(item['metadata']) if item['metadata'] else None
        return item

    def find(exam_id):
        with closing(connect()) as con:
            row = con.execute('SELECT * FROM exams WHERE id = ?', (exam_id,)).fetchone()
        if row is None:
            raise HTTPException(404, 'Examination not found.')
        return exam_record(row)

    def report_for(exam_id):
        exam = find(exam_id)
        if exam['state'] != 'ready':
            raise HTTPException(409, 'Wait for the examination to finish.')
        return json.loads((storage / exam_id / 'report.json').read_text())

    def reviews_for(exam_id):
        with closing(connect()) as con:
            rows = con.execute('SELECT payload FROM reviews WHERE exam_id = ? ORDER BY updated', (exam_id,))
            return [json.loads(row[0]) for row in rows]

    def update(exam_id, state, percentage, stage, error=None):
        with closing(connect()) as con, con:
            con.execute('UPDATE exams SET state=?, progress=?, stage=?, error=? WHERE id=?',
                        (state, percentage, stage, error, exam_id))

    def run(exam_id):
        exam = find(exam_id)
        work = storage / exam_id

        def progress(amount, stage):
            update(exam_id, 'running', amount, stage)

        try:
            progress(.01, 'Opening source video')
            import torch

            mps_ready = torch.backends.mps.is_available()
            device = 'mps' if mps_ready and exam['device'] != 'cpu' else 'cpu'
            if device == 'cpu':
                torch.set_num_threads(4)
            try:
                processor(work / exam['filename'], work, exam['profile'], device,
                          bool(exam['cavities']), progress)
            except (RuntimeError, NotImplementedError) as exc:
                if device != 'mps':
                    raise
                LOG.warning('MPS runtime failed for %s. Retrying on CPU: %s', exam_id, exc)
                progress(.02, 'Apple GPU unavailable for this model. Retrying on CPU')
                torch.set_num_threads(4)
                processor(work / exam['filename'], work, exam['profile'], 'cpu',
                          bool(exam['cavities']), progress)
            update(exam_id, 'ready', 1, 'Annotated recording ready')
        except ValueError as exc:
            update(exam_id, 'failed', 0, 'Analysis stopped', str(exc))
        except Exception:
            LOG.exception('Examination %s failed', exam_id)
            update(exam_id, 'failed', 0, 'Analysis stopped',
                   'The local model or encoder failed. Check the server log and retry.')

    @asynccontextmanager
    async def lifespan(_app):
        storage.mkdir(parents=True, exist_ok=True)
        with closing(connect()) as con, con:
            con.execute('PRAGMA journal_mode=WAL')
            con.executescript("""
                CREATE TABLE IF NOT EXISTS exams (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, filename TEXT NOT NULL,
                    bytes INTEGER NOT NULL, created TEXT NOT NULL, profile TEXT NOT NULL,
                    device TEXT NOT NULL, cavities INTEGER NOT NULL, state TEXT NOT NULL,
                    progress REAL NOT NULL, stage TEXT NOT NULL, error TEXT, metadata TEXT
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    exam_id TEXT NOT NULL, id TEXT NOT NULL, updated TEXT NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY (exam_id, id)
                );
            """)
            columns = {row['name'] for row in con.execute('PRAGMA table_info(exams)')}
            if 'metadata' not in columns:
                con.execute('ALTER TABLE exams ADD COLUMN metadata TEXT')
            con.execute("""UPDATE exams SET state='failed', progress=0, stage='Interrupted',
                        error='The server stopped before analysis finished. Use Retry analysis.'
                        WHERE state IN ('queued', 'running')""")
        yield
        pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title='DentiLens XR', lifespan=lifespan)

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'models_ready': models_ready()}

    @app.get('/api/exams')
    def list_exams():
        with closing(connect()) as con:
            return [exam_record(row) for row in con.execute('SELECT * FROM exams ORDER BY created DESC LIMIT 50')]

    @app.post('/api/exams', status_code=202)
    def create_exam(video: Annotated[UploadFile, File()], profile: Annotated[Profile, Form()] = 'balanced',
                    device: Annotated[Device, Form()] = 'auto', cavities: Annotated[bool, Form()] = False):
        title = Path((video.filename or 'examination.mp4').replace('\\', '/')).name[:200]
        extension = Path(title).suffix.lower()
        if extension not in EXTENSIONS:
            raise HTTPException(415, 'Use an MP4, MOV, M4V or WebM video.')
        if processor is process_exam and not models_ready():
            raise HTTPException(503, 'Install the local tooth model files before analyzing an examination.')
        exam_id = uuid4().hex
        work = storage / exam_id
        work.mkdir()
        filename = f'upload{extension}'
        size = 0
        try:
            with (work / filename).open('wb') as stream:
                while chunk := video.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, 'The local upload limit is 500 MB.')
                    stream.write(chunk)
            if size == 0:
                raise HTTPException(400, 'The uploaded file is empty.')
            try:
                metadata = video_info(work / filename)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            if metadata['duration'] > 600:
                raise HTTPException(422, 'The local limit is 10 minutes per recording.')
            with closing(connect()) as con, con:
                con.execute('INSERT INTO exams VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                            (exam_id, title, filename, size, utcnow(), profile, device, int(cavities),
                             'queued', 0, 'Waiting for the local worker', None, json.dumps(metadata)))
        except BaseException:
            shutil.rmtree(work)
            raise
        pool.submit(run, exam_id)
        return find(exam_id)

    @app.get('/api/exams/{exam_id}')
    def detail(exam_id: UUID):
        result = find(exam_id.hex)
        result['reviews'] = reviews_for(exam_id.hex)
        if result['state'] == 'ready':
            result['report'] = report_for(exam_id.hex)
            if (storage / exam_id.hex / 'reviewed-reel.mp4').is_file():
                result['reel_url'] = f'/api/exams/{exam_id.hex}/media/reviewed-reel.mp4'
        return result

    @app.post('/api/exams/{exam_id}/retry', status_code=202)
    def retry(exam_id: UUID):
        key = exam_id.hex
        find(key)
        with closing(connect()) as con, con:
            updated = con.execute("""UPDATE exams SET state='queued',progress=0,
                                  stage='Waiting for the local worker',error=NULL WHERE id=? AND state='failed'""", (key,))
            if updated.rowcount != 1:
                raise HTTPException(409, 'Only a stopped examination can be retried.')
        pool.submit(run, key)
        return find(key)

    @app.post('/api/exams/{exam_id}/reviews')
    def review(exam_id: UUID, body: Review):
        report = report_for(exam_id.hex)
        data = body.model_dump()
        if body.finding_id:
            finding = next((item for item in report['findings'] if item['id'] == body.finding_id), None)
            if finding is None:
                raise HTTPException(404, 'AI suggestion not found.')
            data['time'] = finding['time']
        if data['time'] >= report['video']['duration']:
            raise HTTPException(422, 'Choose a time within the video.')
        if not body.finding_id and body.decision == 'dismissed':
            raise HTTPException(422, 'Dismiss applies to an AI suggestion.')
        key = body.finding_id or uuid4().hex
        data.update(id=key, updated=utcnow())
        with closing(connect()) as con, con:
            con.execute('INSERT OR REPLACE INTO reviews VALUES (?,?,?,?)',
                        (exam_id.hex, key, data['updated'], json.dumps(data, allow_nan=False)))
        return data

    @app.get('/api/exams/{exam_id}/preview')
    def preview(exam_id: UUID):
        exam = find(exam_id.hex)
        source = storage / exam_id.hex / exam['filename']
        return FileResponse(source)

    @app.get('/api/exams/{exam_id}/media/{asset}')
    def media(exam_id: UUID, asset: Asset, download: bool = False):
        report_for(exam_id.hex)
        path = storage / exam_id.hex / asset
        if not path.is_file():
            raise HTTPException(404, 'Build a reviewed reel first.')
        return FileResponse(path, media_type='video/mp4' if asset.endswith('.mp4') else 'application/x-ndjson',
                            filename=asset if download else None)

    @app.get('/api/exams/{exam_id}/images/{name}')
    def evidence(exam_id: UUID, name: str, download: bool = False):
        report = report_for(exam_id.hex)
        if name not in {item['image'] for item in report['findings']}:
            raise HTTPException(404, 'Evidence image not found.')
        return FileResponse(storage / exam_id.hex / 'images' / name, media_type='image/jpeg',
                            filename=name if download else None)

    @app.get('/api/exams/{exam_id}/evidence.zip')
    def evidence_archive(exam_id: UUID):
        report = report_for(exam_id.hex)
        folder = storage / exam_id.hex / 'images'
        buffer = BytesIO()
        with ZipFile(buffer, 'w') as archive:
            for item in report['findings']:
                archive.write(folder / item['image'], item['image'])
        return Response(buffer.getvalue(), media_type='application/zip',
                        headers={'Content-Disposition': 'attachment; filename=dentilens-evidence.zip'})

    @app.get('/api/exams/{exam_id}/export')
    def export_report(exam_id: UUID):
        exam = find(exam_id.hex)
        report = report_for(exam_id.hex)
        reviews = reviews_for(exam_id.hex)
        return JSONResponse({'examination_id': exam_id.hex, 'title': exam['title'], **report, 'reviewer_notes': reviews},
                            headers={'Content-Disposition': 'attachment; filename=dentilens-examination.json'})

    @app.post('/api/exams/{exam_id}/reel')
    def reel(exam_id: UUID, body: ReelRequest):
        report = report_for(exam_id.hex)
        reviews = {item['id']: item for item in reviews_for(exam_id.hex) if item['decision'] == 'confirmed'}
        if not all(key in reviews for key in body.bookmarks):
            raise HTTPException(422, 'Choose confirmed bookmarks for the reviewed reel.')
        work = storage / exam_id.hex
        media = report['video']
        findings = {item['id']: item for item in report['findings']}
        marks = []
        for item in (reviews[key] for key in dict.fromkeys(body.bookmarks)):
            finding = findings.get(item['id'], {})
            frame = finding.get('frame', min(media['frames'] - 1, round(item['time'] * media['fps'])))
            mark = dict(source_frame=frame, timestamp_sec=frame / media['fps'],
                        model_label=finding.get('label', 'Manual bookmark'),
                        category=finding.get('category'),
                        review=dict(decision='Confirm', tooth=item['label'], note=item['note'], flag=item['flagged']))
            if 'box' in finding:
                mark['bbox_xyxy_normalized'] = [value / extent for value, extent in
                                               zip(finding['box'], [media['width'], media['height']] * 2)]
            marks.append(mark)
        windows = context_windows([item['timestamp_sec'] for item in marks], media['duration'])
        try:
            metadata = export_annotated_reel(work / 'source.mp4', work / 'reviewed-reel.mp4',
                {'fps': media['fps'], 'frames_decoded': media['frames']}, windows, marks, work)
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {'url': f'/api/exams/{exam_id.hex}/media/reviewed-reel.mp4', **metadata}

    if serve_frontend:
        dist = ROOT / 'frontend/dist'
        if (dist / 'index.html').is_file():
            app.mount('/', StaticFiles(directory=dist, html=True), name='workspace')
        else:
            @app.get('/')
            def setup_notice():
                return {'setup': 'Build the frontend with npm --prefix frontend run build.'}

    return app


app = create_app()
