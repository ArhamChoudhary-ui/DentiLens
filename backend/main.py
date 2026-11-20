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

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
import cv2
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from annotated_reel import context_windows, export_annotated_reel
from full_video import process_exam, stamp, video_info
from exam_intelligence import evidence_index, quality_report, search_exam, tooth_history
from reviewed_video import confirmed_row, render_reviewed

LOG = logging.getLogger('dentilens')
MAX_BYTES = 500 * 1024 * 1024
EXTENSIONS = {'.mp4', '.mov', '.m4v', '.webm'}
Profile = Literal['fast', 'balanced', 'detailed']
Device = Literal['auto', 'cpu', 'mps']
Asset = Literal['source.mp4', 'annotated.mp4', 'reviewed-reel.mp4', 'timeline.jsonl', 'confirmed-export.mp4', 'suggestions-export.mp4']


class Review(BaseModel):
    time: float = Field(ge=0, allow_inf_nan=False)
    finding_id: str | None = Field(default=None, pattern=r'^ai-\d{2}$')
    label: str = Field(default='', max_length=100)
    note: str = Field(default='', max_length=500)
    decision: Literal['confirmed', 'dismissed'] = 'confirmed'
    flagged: bool = False
    fdi_number: str | None = Field(default=None, pattern=r'^[1-4][1-8]$')
    box: tuple[Annotated[float, Field(ge=0, le=1)], Annotated[float, Field(ge=0, le=1)],
               Annotated[float, Field(ge=0, le=1)], Annotated[float, Field(ge=0, le=1)]] | None = None

    @model_validator(mode='after')
    def check_box(self):
        if self.box and (self.box[0] >= self.box[2] or self.box[1] >= self.box[3]):
            raise ValueError('The region must have a positive width and height.')
        return self


class VideoExport(BaseModel):
    version: Literal['suggestions', 'confirmed'] = 'confirmed'
    boxes: bool = True
    labels: bool = True
    fdi: bool = True
    timestamp: bool = True
    scores: bool = False
    track_ids: bool = False


class AnalysisCancelled(Exception):
    pass


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
            con.execute("UPDATE exams SET state=CASE WHEN state='canceling' THEN 'cancelled' ELSE ? END, "
                        "progress=?, stage=CASE WHEN state='canceling' THEN 'Analysis cancelled' ELSE ? END, "
                        "error=CASE WHEN state='canceling' THEN NULL ELSE ? END WHERE id=?",
                        (state, percentage, stage, error, exam_id))

    def run(exam_id):
        exam = find(exam_id)
        work = storage / exam_id

        def progress(amount, stage):
            with closing(connect()) as con, con:
                updated = con.execute("UPDATE exams SET state='running', progress=?, stage=? WHERE id=? "
                                      "AND state IN ('queued', 'running')", (amount, stage, exam_id))
                if not updated.rowcount:
                    raise AnalysisCancelled

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
            with closing(connect()) as con, con:
                changed = con.execute("UPDATE exams SET state='ready',progress=1,stage='Annotated recording ready' "
                                      "WHERE id=? AND state='running'", (exam_id,))
                if not changed.rowcount:
                    raise AnalysisCancelled
        except AnalysisCancelled:
            update(exam_id, 'cancelled', 0, 'Analysis cancelled')
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
                CREATE TABLE IF NOT EXISTS review_history (
                    exam_id TEXT NOT NULL, id TEXT NOT NULL, updated TEXT NOT NULL, payload TEXT NOT NULL
                );
            """)
            columns = {row['name'] for row in con.execute('PRAGMA table_info(exams)')}
            if 'metadata' not in columns:
                con.execute('ALTER TABLE exams ADD COLUMN metadata TEXT')
            con.execute("""UPDATE exams SET state='failed', progress=0, stage='Interrupted',
                        error='The server stopped before analysis finished. Use Retry analysis.'
                        WHERE state IN ('queued', 'running')""")
            con.execute("UPDATE exams SET state='cancelled',progress=0,stage='Analysis cancelled' WHERE state='canceling'")
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
                    device: Annotated[Device, Form()] = 'auto', cavities: Annotated[bool, Form()] = False,
                    name: Annotated[str, Form(max_length=100)] = ''):
        original = Path((video.filename or 'examination.mp4').replace('\\', '/')).name[:200]
        title = name.strip() or original
        extension = Path(original).suffix.lower()
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

    @app.delete('/api/exams/{exam_id}', status_code=204)
    def delete_exam(exam_id: UUID):
        key = exam_id.hex
        with closing(connect()) as con, con:
            con.execute('BEGIN IMMEDIATE')
            current = con.execute('SELECT state FROM exams WHERE id=?', (key,)).fetchone()
            if not current:
                raise HTTPException(404, 'Examination not found.')
            if current['state'] in ('queued', 'running', 'canceling'):
                raise HTTPException(409, 'Stop analysis before deleting this examination.')
            shutil.rmtree(storage / key)
            for table in ('review_history', 'reviews'):
                con.execute(f'DELETE FROM {table} WHERE exam_id=?', (key,))
            con.execute('DELETE FROM exams WHERE id=?', (key,))
        return Response(status_code=204)

    @app.post('/api/exams/{exam_id}/retry', status_code=202)
    def retry(exam_id: UUID):
        key = exam_id.hex
        find(key)
        with closing(connect()) as con, con:
            updated = con.execute("""UPDATE exams SET state='queued',progress=0,
                                  stage='Waiting for the local worker',error=NULL WHERE id=? AND state IN ('failed', 'cancelled')""", (key,))
            if updated.rowcount != 1:
                raise HTTPException(409, 'Only a stopped examination can be retried.')
        pool.submit(run, key)
        return find(key)

    @app.post('/api/exams/{exam_id}/cancel', status_code=202)
    def cancel(exam_id: UUID):
        key = exam_id.hex
        find(key)
        with closing(connect()) as con, con:
            updated = con.execute("UPDATE exams SET state='canceling',stage='Stopping analysis' "
                                  "WHERE id=? AND state IN ('queued', 'running')", (key,))
            if not updated.rowcount:
                raise HTTPException(409, 'This examination is no longer running.')
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
            data['original_ai'] = {field: finding.get(field) for field in ('label', 'fdi_number', 'box', 'track_id')}
        if data['time'] >= report['video']['duration']:
            raise HTTPException(422, 'Choose a time within the video.')
        if not body.finding_id and body.decision == 'dismissed':
            raise HTTPException(422, 'Dismiss applies to an AI suggestion.')
        key = body.finding_id or uuid4().hex
        data.update(id=key, updated=utcnow())
        with closing(connect()) as con, con:
            record = (exam_id.hex, key, data['updated'], json.dumps(data, allow_nan=False))
            con.execute('INSERT OR REPLACE INTO reviews VALUES (?,?,?,?)', record)
            con.execute('INSERT INTO review_history VALUES (?,?,?,?)', record)
        return data

    @app.get('/api/exams/{exam_id}/tooth-history')
    def history(exam_id: UUID, fdi: Annotated[str | None, Query(pattern=r'^[1-4][1-8]$')] = None,
                track: Annotated[str | None, Query(pattern=r'^T\d+-\d+$')] = None):
        if not fdi and not track:
            raise HTTPException(422, 'Choose an FDI number or tooth track.')
        key = exam_id.hex
        report = report_for(key)
        return tooth_history(storage / key / 'timeline.jsonl', report, reviews_for(key), fdi, track)

    @app.get('/api/exams/{exam_id}/intelligence')
    def intelligence(exam_id: UUID, q: Annotated[str, Query(min_length=1, max_length=160)]):
        key = exam_id.hex
        report = report_for(key)
        return search_exam(storage / key / 'timeline.jsonl', report, reviews_for(key), q.strip())

    @app.get('/api/exams/{exam_id}/evidence-index')
    def export_evidence_index(exam_id: UUID):
        key = exam_id.hex
        report = report_for(key)
        data = evidence_index(storage / key / 'timeline.jsonl', report, reviews_for(key))
        return JSONResponse({'examination_id': key, **data},
                            headers={'Content-Disposition': 'attachment; filename=dentilens-evidence-index.json'})

    @app.get('/api/exams/{exam_id}/quality')
    def quality(exam_id: UUID):
        key = exam_id.hex
        report = report_for(key)
        try:
            return quality_report(storage / key / 'source.mp4', report['video'])
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get('/api/exams/{exam_id}/preview')
    @app.head('/api/exams/{exam_id}/preview', include_in_schema=False)
    def preview(exam_id: UUID):
        exam = find(exam_id.hex)
        source = storage / exam_id.hex / exam['filename']
        return FileResponse(source)

    @app.get('/api/exams/{exam_id}/media/{asset}')
    @app.head('/api/exams/{exam_id}/media/{asset}', include_in_schema=False)
    def media(exam_id: UUID, asset: Asset, download: bool = False):
        report_for(exam_id.hex)
        path = storage / exam_id.hex / asset
        if not path.is_file():
            raise HTTPException(404, 'Build this export first.')
        return FileResponse(path, media_type='video/mp4' if asset.endswith('.mp4') else 'application/x-ndjson',
                            filename=asset if download else None)

    @app.get('/api/exams/{exam_id}/images/{name}')
    @app.head('/api/exams/{exam_id}/images/{name}', include_in_schema=False)
    def evidence(exam_id: UUID, name: str, download: bool = False):
        report = report_for(exam_id.hex)
        if name not in {item['image'] for item in report['findings']}:
            raise HTTPException(404, 'Evidence image not found.')
        return FileResponse(storage / exam_id.hex / 'images' / name, media_type='image/jpeg',
                            filename=name if download else None)

    @app.get('/api/exams/{exam_id}/still.jpg')
    def still(exam_id: UUID, time: Annotated[float, Query(ge=0, allow_inf_nan=False)] = 0,
              bookmark: str | None = None):
        report = report_for(exam_id.hex)
        info = report['video']
        review = None
        if bookmark:
            review = next((item for item in reviews_for(exam_id.hex)
                           if item['id'] == bookmark and item['decision'] == 'confirmed'), None)
            if not review:
                raise HTTPException(404, 'Confirmed bookmark not found.')
            time = review['time']
        if time >= info['duration']:
            raise HTTPException(422, 'Choose a time within the video.')
        frame_index = min(info['frames'] - 1, round(time * info['fps']))
        capture = cv2.VideoCapture(str(storage / exam_id.hex / 'source.mp4'))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise HTTPException(422, 'Could not load this frame.')
        if review:
            finding = next((item for item in report['findings'] if item['id'] == review['finding_id']), None)
            if finding or review.get('box'):
                row = confirmed_row(finding or {'category': 'tooth_identity', 'confidence': 0}, review,
                                    finding, frame_index, info['width'], info['height'])
                frame = stamp(frame, time, [row], {'version': 'confirmed', 'fdi': True})
        encoded, payload = cv2.imencode('.jpg', frame)
        if not encoded:
            raise HTTPException(422, 'Could not save this frame.')
        return Response(payload.tobytes(), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

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
        with closing(connect()) as con:
            history = [json.loads(row[0]) for row in con.execute(
                'SELECT payload FROM review_history WHERE exam_id=? ORDER BY rowid', (exam_id.hex,))]
        return JSONResponse({'examination_id': exam_id.hex, 'title': exam['title'], **report,
                             'reviewer_notes': reviews, 'review_history': history},
                            headers={'Content-Disposition': 'attachment; filename=dentilens-examination.json'})

    @app.post('/api/exams/{exam_id}/video-export')
    def video_export(exam_id: UUID, body: VideoExport):
        report = report_for(exam_id.hex)
        reviews = reviews_for(exam_id.hex)
        if body.version == 'confirmed' and not any(item['decision'] == 'confirmed' and (item.get('finding_id') or item.get('box')) for item in reviews):
            raise HTTPException(422, 'Confirm a bookmark with a region before exporting.')
        try:
            result = render_reviewed(storage / exam_id.hex, report, reviews, body.model_dump())
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {'url': f'/api/exams/{exam_id.hex}/media/{result.name}'}

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
                        review=dict(decision='Confirm', tooth=' · '.join(filter(None, [item['label'], item.get('fdi_number')])),
                                    note=item['note'], flag=item['flagged']))
            if item.get('box'):
                mark['bbox_xyxy_normalized'] = item['box']
            elif 'box' in finding:
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
