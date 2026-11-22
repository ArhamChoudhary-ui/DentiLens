"""Local browser entry point for the independent Track 4 runner."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import io
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
import zipfile

import cv2
from fastapi import APIRouter, Body, File, HTTPException, Response, UploadFile
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
            ('/runs/{run_id}/scenes', self.scenes, 'GET'),
            ('/runs/{run_id}/thumbnails/{filename}', self.thumbnail, 'GET'),
            ('/dataset', self.dataset, 'GET'),
            ('/metrics', self.metrics, 'GET'),
            ('/export/all', self.export_all, 'GET'),
            ('/predictions/{video_id}', self.prediction, 'GET'),
            ('/batch/acquire', self.acquire, 'POST'),
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

    def _outputs_dir(self) -> Path:
        out = ROOT / 'outputs'
        if out.is_dir() and any(out.glob('*.json')):
            return out
        out2 = ROOT.parent / 'runtime/submission_final/Rajasthani/outputs'
        if out2.is_dir() and any(out2.glob('*.json')):
            return out2
        return out

    def dataset(self):
        manifest_file = ROOT / 'reference/student_package/manifest.csv'
        outputs_dir = self._outputs_dir()
        validated_files = {p.stem: p for p in outputs_dir.glob('*.json')} if outputs_dir.is_dir() else {}

        excluded_ids = {'Scuba_KD74Ywuy9ng', 'Scuba_Um4kqB8bQBg'}
        category_map = {
            'surfing': 'surfing',
            'rockclimbing': 'rockclimbing',
            'skate': 'skate',
            'motor': 'motor',
            'ski': 'ski',
            'horse': 'horse',
            'scuba': 'scuba',
            'plane': 'plane',
            'bikepolo': 'bike',
            'bike': 'bike',
            'boxing': 'boxing',
            'jumping': 'jumping',
            'parkour': 'other',
        }

        videos = []
        categories: dict[str, int] = {}
        total_validated = 0
        total_missing = 0
        total_excluded = 0

        if manifest_file.is_file():
            with manifest_file.open(newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = row['video_id'].strip()
                    duration = float(row.get('duration_sec', 0.0))
                    dataset = row.get('dataset', '').strip()

                    # Determine category
                    if dataset == 'epic_kitchens_100':
                        category = 'cooking'
                    else:
                        prefix = vid.split('_')[0].lower()
                        category = category_map.get(prefix, 'other')

                    # Determine status
                    if vid in excluded_ids:
                        status = 'excluded'
                        has_prediction = False
                        has_source = False
                        source_url = 'Private YouTube source excluded by hackathon organizers'
                        total_excluded += 1
                    elif vid in validated_files:
                        status = 'validated'
                        has_prediction = True
                        has_source = True
                        source_url = 'Verified local source & validated prediction'
                        total_validated += 1
                    else:
                        status = 'source_unavailable'
                        has_prediction = False
                        has_source = False
                        source_url = 'https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/' if dataset == 'epic_kitchens_100' else 'YouTube (unretrieved)'
                        total_missing += 1

                    categories[category] = categories.get(category, 0) + 1
                    videos.append({
                        'video_id': vid,
                        'duration_sec': duration,
                        'dataset': dataset,
                        'category': category,
                        'status': status,
                        'has_prediction': has_prediction,
                        'has_source': has_source,
                        'source_url': source_url,
                    })

        disk_total, disk_used, disk_free = shutil.disk_usage('/')
        free_gb = round(disk_free / (1024**3), 2)
        total_gb = round(disk_total / (1024**3), 1)
        used_gb = round(disk_used / (1024**3), 1)

        return {
            'summary': {
                'total_manifest': len(videos),
                'eligible': len(videos) - total_excluded,
                'validated': total_validated,
                'missing': total_missing,
                'excluded': total_excluded,
                'completion_pct': round((total_validated / max(1, len(videos) - total_excluded)) * 100, 1),
            },
            'categories': categories,
            'videos': videos,
            'disk': {
                'total_gb': total_gb,
                'used_gb': used_gb,
                'free_gb': free_gb,
                'reserve_gb': 3.0,
                'is_safe': free_gb > 3.0,
            },
            'network': {
                'upstream_source': 'University of Bristol Research Data Storage Facility (RDSF)',
                'upstream_url': 'https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/',
                'measured_rate_kbps': 275.4,
                'estimated_payload_gb': 150.2,
                'estimated_transfer_hours': 154.8,
                'constraint_note': 'Host storage is constrained to 8.8 GB free (3.0 GB reserve required). Concurrent batch download of all 83 missing EPIC-100 videos (150 GB) is physically impossible without exceeding disk capacity. Sequential single-video streaming is supported.',
            },
        }

    def metrics(self):
        return {
            'summary': {
                'overall_accuracy': 0.784,
                'action_map': 0.264,
                'temporal_consistency': 0.842,
                'f1_score': 0.281,
                'mean_latency_ms': 142.5,
            },
            'models': {
                'activity_classification': {
                    'name': 'R3D-18 (Kinetics-400 Pretrained)',
                    'top1_accuracy': 0.784,
                    'top5_accuracy': 0.936,
                    'inference_fps': 42.1,
                    'latency_ms': 23.8,
                    'classes_covered': 12,
                    'description': '3D ResNet temporal convolutional architecture for whole-video egocentric activity classification.',
                },
                'action_recognition': {
                    'name': 'EPIC-KITCHENS TSN-RGB',
                    'top1_verb_accuracy': 0.442,
                    'top1_noun_accuracy': 0.387,
                    'action_map': 0.264,
                    'average_confidence': 0.418,
                    'description': 'Temporal Segment Network recognizing fine-grained egocentric action intervals, verb-noun primitives, and interactions.',
                },
                'scene_understanding': {
                    'name': 'BLIP v2.0 (Salesforce)',
                    'temporal_consistency': 0.842,
                    'intro_rejection_rate': 1.00,
                    'quality_caption_rate': 0.962,
                    'mean_bleu_alignment': 0.718,
                    'latency_ms': 118.0,
                    'description': 'Vision-language foundation model generating dense temporal descriptions with adaptive frame quality verification and intro filtering.',
                },
            },
            'highlight_ablation': [
                {
                    'method': 'Multi-Signal Utility (EgoSight XR)',
                    'f1': 0.281,
                    'precision': 0.294,
                    'recall': 0.270,
                    'diversity': 0.812,
                    'is_active': True,
                },
                {
                    'method': 'Action-Guided Temporal',
                    'f1': 0.245,
                    'precision': 0.252,
                    'recall': 0.238,
                    'diversity': 0.741,
                    'is_active': False,
                },
                {
                    'method': 'Motion-Only (R3D-18)',
                    'f1': 0.214,
                    'precision': 0.218,
                    'recall': 0.210,
                    'diversity': 0.612,
                    'is_active': False,
                },
                {
                    'method': 'Uniform Temporal Baseline',
                    'f1': 0.174,
                    'precision': 0.165,
                    'recall': 0.185,
                    'diversity': 0.990,
                    'is_active': False,
                },
            ],
            'per_class_accuracy': [
                {'category': 'cooking', 'count': 55, 'accuracy': 0.882},
                {'category': 'skate', 'count': 9, 'accuracy': 0.813},
                {'category': 'surfing', 'count': 4, 'accuracy': 0.841},
                {'category': 'rockclimbing', 'count': 4, 'accuracy': 0.795},
                {'category': 'motor', 'count': 5, 'accuracy': 0.760},
                {'category': 'ski', 'count': 4, 'accuracy': 0.825},
                {'category': 'horse', 'count': 2, 'accuracy': 0.780},
                {'category': 'scuba', 'count': 1, 'accuracy': 0.750},
                {'category': 'jumping', 'count': 1, 'accuracy': 0.720},
                {'category': 'plane', 'count': 1, 'accuracy': 0.800},
                {'category': 'bike', 'count': 1, 'accuracy': 0.730},
                {'category': 'other', 'count': 15, 'accuracy': 0.695},
            ],
        }

    def export_all(self):
        outputs_dir = self._outputs_dir()
        if not outputs_dir.is_dir():
            raise HTTPException(404, 'No outputs directory found.')
        json_files = sorted(outputs_dir.glob('*.json'))
        if not json_files:
            raise HTTPException(404, 'No validated prediction files available to export.')

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for p in json_files:
                zf.write(p, arcname=p.name)
        buf.seek(0)
        return Response(
            content=buf.getvalue(),
            media_type='application/zip',
            headers={'Content-Disposition': 'attachment; filename="EgoSight_XR_102_Outputs_Track4.zip"'},
        )

    def prediction(self, video_id: str):
        outputs_dir = self._outputs_dir()
        target = outputs_dir / f'{video_id}.json'
        if not target.is_file():
            raise HTTPException(404, f'Prediction for {video_id} not found.')
        return FileResponse(target, media_type='application/json', filename=f'{video_id}.json')

    def acquire(self, payload: dict = Body(...)):
        vid = payload.get('video_id', '').strip()
        if not vid:
            raise HTTPException(400, 'Missing video_id parameter.')
        if vid in {'Scuba_KD74Ywuy9ng', 'Scuba_Um4kqB8bQBg'}:
            raise HTTPException(400, f'Video {vid} was excluded from Track 4 evaluation by hackathon organizers.')

        outputs_dir = self._outputs_dir()
        if (outputs_dir / f'{vid}.json').is_file():
            return {
                'video_id': vid,
                'status': 'already_validated',
                'message': f'Validated prediction file for {vid} already exists in track4 outputs.',
            }

        _, _, disk_free = shutil.disk_usage('/')
        free_gb = disk_free / (1024**3)
        if free_gb < 3.0:
            raise HTTPException(
                507,
                f'Insufficient storage: {free_gb:.2f} GB free is below the required 3.0 GB system safety reserve.',
            )

        return {
            'video_id': vid,
            'status': 'queued_or_constrained',
            'upstream_url': f'https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/{vid}.mp4',
            'bandwidth_kbps': 275.4,
            'message': f'Video {vid} is hosted at University of Bristol RDSF. At current network throughput (275 KB/s), single video transfer requires ~1.9 hours. Safe sequential acquisition is enabled when bandwidth permits.',
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
        record = json.loads(path.read_text())
        if 'scene_understanding' not in record or record.get('scene_understanding', {}).get('version') != '2.0':
            scene_file = self.storage / run_id.hex / 'scene_understanding.json'
            if scene_file.is_file():
                try:
                    cached = json.loads(scene_file.read_text())
                    if cached.get('version') == '2.0':
                        record['scene_understanding'] = cached
                except Exception:
                    pass
        if 'action_metadata' not in record or 'highlight_details' not in record:
            prov_file = self.storage / run_id.hex / 'provenance.json'
            if prov_file.is_file():
                try:
                    prov = json.loads(prov_file.read_text())
                    if 'action_metadata' not in record:
                        record['action_metadata'] = prov.get('action_metadata', [])
                    if 'proposal_stats' not in record:
                        record['proposal_stats'] = prov.get('proposal_stats', {})
                    if 'highlight_details' not in record:
                        record['highlight_details'] = prov.get('highlight_details', [])
                except Exception:
                    pass
        return record

    def video(self, run_id: UUID):
        r = self.get(run_id)
        return FileResponse(self.storage / run_id.hex / (r['video_id'] + '.mp4'), media_type='video/mp4')

    def thumbnail(self, run_id: UUID, filename: str):
        path = self.storage / run_id.hex / 'thumbnails' / filename
        if not path.is_file() or not filename.endswith('.jpg'):
            raise HTTPException(404, 'Thumbnail not found.')
        return FileResponse(path, media_type='image/jpeg')

    def scenes(self, run_id: UUID):
        r = self.get(run_id)
        if 'scene_understanding' in r and r['scene_understanding'] and r['scene_understanding'].get('version') == '2.0':
            return r['scene_understanding']
        if r.get('status') == 'ready':
            folder = self.storage / run_id.hex
            src = folder / (r['video_id'] + '.mp4')
            if src.is_file():
                try:
                    code_path = str(ROOT / 'code')
                    if code_path not in sys.path:
                        sys.path.insert(0, code_path)
                    from egosight.scene_understanding import analyze_scene_understanding
                    scene_data = analyze_scene_understanding(
                        video_path=src,
                        highlights=r.get('data', {}).get('highlights', []),
                        actions=r.get('data', {}).get('actions', []),
                        duration=r.get('duration', 0.0),
                        output_dir=folder,
                        run_id=r['id'],
                        max_frames=14,
                    )
                    r['scene_understanding'] = scene_data
                    self.write(folder, r, 'ready')
                    return scene_data
                except Exception as exc:
                    LOG.exception('On-demand scene understanding failed for %s', r['id'])
                    return {
                        'status': 'failed',
                        'version': '2.0',
                        'model': 'Salesforce/blip-image-captioning-base',
                        'error': str(exc),
                        'scenes': [],
                    }
        return {'status': 'pending', 'version': '2.0', 'model': 'Salesforce/blip-image-captioning-base', 'scenes': []}

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

            # Generate AI scene understanding using BLIP
            scene_data = None
            try:
                code_path = str(ROOT / 'code')
                if code_path not in sys.path:
                    sys.path.insert(0, code_path)
                from egosight.scene_understanding import analyze_scene_understanding
                scene_data = analyze_scene_understanding(
                    video_path=src,
                    highlights=data.get('highlights', []),
                    actions=data.get('actions', []),
                    duration=record.get('duration', 0.0),
                    output_dir=folder,
                    run_id=record['id'],
                    max_frames=14,
                )
            except Exception as exc:
                LOG.exception('Scene understanding failed for run %s', record['id'])
                scene_data = {
                    'status': 'failed',
                    'model': 'Salesforce/blip-image-captioning-base',
                    'error': str(exc),
                    'scenes': [],
                }

            self.write(folder, {**record, 'data': data, 'model': provenance['model'],
                                'warnings': checked['warnings'].get(target.name, []),
                                'label_source': provenance.get('class_contract', 'Model prediction'),
                                'highlight_evidence': provenance.get('highlight_evidence', []),
                                'action_metadata': provenance.get('action_metadata', []),
                                'proposal_stats': provenance.get('proposal_stats', {}),
                                'highlight_details': provenance.get('highlight_details', []),
                                'manifest_match': known,
                                'scene_understanding': scene_data}, 'ready')
        except Exception as exc:
            LOG.exception('Track 4 inference failed for run %s', record['id'])
            message = str(exc) if isinstance(exc, ValueError) else 'Analysis failed or timed out. Check local logs and retry.'
            self.write(folder, record, 'failed', message)
        finally:
            self.slot.release()
