"""Local browser entry point for the independent Track 4 runner."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from threading import Lock
from typing import Annotated
from uuid import UUID, uuid4

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1] / 'track4'
MAX_BYTES = 500 * 1024 * 1024
LOG = logging.getLogger(__name__)


class ActionWorkspace:
    def __init__(self, storage: Path):
        self.storage = storage
        self.slot = Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='egosight')
        self.router = APIRouter(prefix='/api/action')
        for route, endpoint, method in [
            ('/settings', self.settings, 'GET'),
            ('/runs', self.history, 'GET'),
            ('/runs', self.upload, 'POST'),
            ('/runs/{run_id}', self.get, 'GET'),
            ('/runs/{run_id}/video', self.video, 'GET'),
            ('/runs/{run_id}/result', self.result, 'GET'),
        ]:
            self.router.add_api_route(route, endpoint, methods=[method])

    def settings(self):
        kit = ROOT / 'reference/student_package'
        snapshot = ROOT.parent / 'submission/EgoSight XR/Batch Results.json'
        return {
            'benchmark': json.loads(snapshot.read_text()) if snapshot.is_file() else None,
            'models': {
                'EPIC-KITCHENS': (ROOT / 'weights/tsn_rgb.ckpt').is_file()
                and (ROOT / 'weights/epic-classes.json').is_file(),
                'FPVSum': (ROOT / 'weights/r3d_18-b3b3357e.pth').is_file(),
            },
            'labels': json.loads((kit / 'supported_classes.json').read_text())['labels'],
            'max_bytes': MAX_BYTES,
        }

    def restore(self):
        for path in self.storage.glob('*/run.json'):
            record = json.loads(path.read_text())
            if record['status'] in {'running', 'validating'}:
                self.write(path.parent, record, 'failed', 'The server stopped before inference completed.')

    @staticmethod
    def write(folder, record, status, error=None):
        updated = {**record, 'status': status, 'error': error}
        temp = folder / 'run.tmp'
        temp.write_text(json.dumps(updated))
        temp.replace(folder / 'run.json')

    def history(self):
        files = sorted(self.storage.glob('*/run.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        return [json.loads(f.read_text()) for f in files[:50]]

    def get(self, run_id: UUID):
        path = self.storage / run_id.hex / 'run.json'
        if not path.is_file():
            raise HTTPException(404, 'Analysis not found.')
        return json.loads(path.read_text())

    def video(self, run_id: UUID):
        r = self.get(run_id)
        return FileResponse(self.storage / run_id.hex / (r['video_id'] + '.mp4'), media_type='video/mp4')

    def result(self, run_id: UUID):
        r = self.get(run_id)
        if r['status'] != 'ready':
            raise HTTPException(409, 'Validated predictions are not available yet.')
        return FileResponse(self.storage / run_id.hex / (r['video_id'] + '.json'),
                            media_type='application/json', filename=r['video_id'] + '.json')

    def upload(self, video: Annotated[UploadFile, File()]):
        name = Path(video.filename or '').name
        if Path(name).suffix.lower() != '.mp4':
            raise HTTPException(415, 'Choose an MP4 video.')
        if not self.slot.acquire(blocking=False):
            raise HTTPException(409, 'Wait for the current video analysis to finish.')
        key = uuid4().hex
        folder = self.storage / key
        accepted = False
        try:
            folder.mkdir(parents=True)
            safe = re.sub(r'[^A-Za-z0-9_-]+', '-', Path(name).stem).strip('-_')[:100] or 'video'
            src = folder / (safe + '.mp4')
            size = 0
            with src.open('wb') as out:
                while chunk := video.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, 'Choose an MP4 smaller than 500 MB.')
                    out.write(chunk)
            cap = cv2.VideoCapture(str(src))
            try:
                frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
                if not (math.isfinite(frames) and math.isfinite(fps) and fps > 0):
                    raise HTTPException(422, 'The MP4 must contain readable video metadata.')
                if not cap.read()[0] or not frames / fps >= 4:
                    raise HTTPException(422, 'The MP4 must be readable and at least four seconds long.')
                duration = frames / fps
            finally:
                cap.release()
            record = {'id': key, 'name': name, 'video_id': safe, 'bytes': size,
                      'duration': duration, 'created': datetime.now(timezone.utc).isoformat()}
            self.write(folder, record, 'running')
            self.pool.submit(self.process, folder, src, record)
            accepted = True
            return self.get(UUID(hex=key))
        finally:
            video.file.close()
            if not accepted:
                shutil.rmtree(folder, ignore_errors=True)
                self.slot.release()

    def process(self, folder, src, record):
        target = src.with_suffix('.json')
        report = folder / 'provenance.json'
        try:
            launch = [sys.executable, str(ROOT / 'code/main.py'), str(src), str(target),
                      '--team-id', 'DEVELOPMENT', '--device', 'cpu', '--report', str(report)]
            manifest = ROOT / 'reference/student_package/manifest.csv'
            known = any(row.startswith(src.stem + ',') for row in manifest.read_text().splitlines()[1:])
            if not known:
                launch += ['--development-video', '--dataset', 'fpvsum']
            finished = subprocess.run(launch, capture_output=True, text=True, timeout=1800)
            if finished.returncode:
                detail = next((line.removeprefix('ERROR: ') for line in reversed(finished.stderr.splitlines())
                               if line.startswith('ERROR: ')), 'Inference failed. Check the local model setup.')
                raise ValueError(detail.replace(str(folder), 'upload').replace(str(ROOT), 'track4'))
            self.write(folder, record, 'validating')
            validation = [sys.executable, str(ROOT / 'scripts/check_submission.py'), '--file', str(target),
                          '--team-id', 'DEVELOPMENT']
            if not known:
                validation.append('--schema-only')
            checked = json.loads(subprocess.check_output(validation, text=True, timeout=60))
            data = json.loads(target.read_text())
            provenance = json.loads(report.read_text())
            self.write(folder, {**record, 'data': data, 'model': provenance['model'],
                                'warnings': checked['warnings'].get(target.name, []),
                                'label_source': provenance.get('class_contract', 'Model prediction'),
                                'highlight_evidence': provenance.get('highlight_evidence', []),
                                'manifest_match': known}, 'ready')
        except Exception as exc:
            LOG.exception('Track 4 inference failed for run %s', record['id'])
            message = str(exc) if isinstance(exc, ValueError) else 'Analysis failed or timed out. Check local logs and retry.'
            self.write(folder, record, 'failed', message)
        finally:
            self.slot.release()
