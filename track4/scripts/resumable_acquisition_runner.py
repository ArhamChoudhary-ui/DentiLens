"""Resumable, storage-aware dataset acquisition and inference pipeline for EgoSight XR."""
import argparse
import csv
from datetime import datetime, timezone
from hashlib import md5
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

try:
    import cv2
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
if str(ROOT / 'code') not in sys.path:
    sys.path.insert(0, str(ROOT / 'code'))
LOG = logging.getLogger('resumable_runner')

GIB = 1024 ** 3
DEFAULT_RESERVE_GIB = 3.0
DEFAULT_RESERVE_BYTES = int(DEFAULT_RESERVE_GIB * GIB)
FREE_RESERVE_GIB = DEFAULT_RESERVE_GIB

VALID_STATUSES = {
    'pending',
    'downloading',
    'downloaded',
    'processing',
    'validated',
    'failed',
    'unavailable',
}


def atomic_write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def check_storage_safety(work_dir: Path, needed_bytes: int, reserve_bytes: int = DEFAULT_RESERVE_BYTES):
    try:
        usage = shutil.disk_usage(work_dir)
        free_bytes = usage.free
        free_gb = round(free_bytes / GIB, 2)
        headroom = free_bytes - reserve_bytes
        if headroom < needed_bytes:
            needed_gb = round(needed_bytes / GIB, 2)
            reserve_gb = round(reserve_bytes / GIB, 2)
            msg = (
                f'Insufficient storage: {free_gb} GB free with {reserve_gb} GB reserve floor. '
                f'Available headroom ({round(headroom / GIB, 2)} GB) is less than needed ({needed_gb} GB).'
            )
            return False, free_gb, msg
        return True, free_gb, 'Storage capacity verified safe.'
    except Exception as exc:
        return False, 0.0, f'Storage check failed: {exc}'


def verify_media_file(path: Path, expected_md5: str = None, check_opencv: bool = True):
    if not path.is_file():
        return False, f'File does not exist: {path}'
    if path.stat().st_size == 0:
        return False, 'File is empty (0 bytes).'

    if expected_md5:
        hasher = md5()
        with path.open('rb') as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest.lower() != expected_md5.lower():
            return False, f'Checksum mismatch: expected {expected_md5}, got {digest}'

    if check_opencv and cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                return False, 'OpenCV could not open video container.'
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if frames <= 0 or fps <= 0:
                return False, f'Invalid video stream metadata: frames={frames}, fps={fps}'
            ret, _ = cap.read()
            if not ret:
                return False, 'Failed to read initial video frame.'
        finally:
            cap.release()

    return True, None


def get_upstream_file_size(url: str, timeout: int = 25) -> int:
    cmd = ['curl', '-fsSL', '-I', '--connect-timeout', '15', '--max-time', str(timeout), url]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f'HTTP HEAD failed for {url}: {res.stderr.strip()}')
    lengths = re.findall(r'^Content-Length:\s*(\d+)', res.stdout, re.IGNORECASE | re.MULTILINE)
    if not lengths:
        raise RuntimeError(f'Upstream server did not supply Content-Length header for {url}')
    return int(lengths[-1])


class AcquisitionState:
    def __init__(self, state_file: Path, team_id: str, eligible_keys: list[str]):
        self.state_file = state_file
        self.team_id = team_id
        self.data = {
            'team_id': team_id,
            'version': '1.0',
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'active_job': None,
            'summary': {
                'total_eligible': len(eligible_keys),
                'validated': 0,
                'pending': len(eligible_keys),
                'downloading': 0,
                'processing': 0,
                'failed': 0,
                'unavailable': 0,
            },
            'videos': {
                k: {
                    'status': 'pending',
                    'attempts': 0,
                    'error': None,
                    'stage': None,
                    'updated': None,
                }
                for k in eligible_keys
            },
        }
        self.load()

    def load(self):
        if self.state_file.is_file():
            try:
                disk_data = json.loads(self.state_file.read_text())
                if disk_data.get('team_id') == self.team_id:
                    for k, v in disk_data.get('videos', {}).items():
                        if k in self.data['videos']:
                            self.data['videos'][k].update(v)
                    self._recalc_summary()
            except Exception as e:
                LOG.warning('Could not load existing state file %s: %e', self.state_file, e)

    def _recalc_summary(self):
        counts = {st: 0 for st in VALID_STATUSES}
        for v in self.data['videos'].values():
            st = v.get('status', 'pending')
            counts[st] = counts.get(st, 0) + 1
        self.data['summary'] = {
            'total_eligible': len(self.data['videos']),
            'validated': counts.get('validated', 0),
            'pending': counts.get('pending', 0),
            'downloading': counts.get('downloading', 0),
            'processing': counts.get('processing', 0),
            'failed': counts.get('failed', 0),
            'unavailable': counts.get('unavailable', 0),
        }
        self.data['last_updated'] = datetime.now(timezone.utc).isoformat()

    def save(self):
        self._recalc_summary()
        atomic_write_json(self.state_file, self.data)

    def update(self, video_id: str, status: str, error: str = None, stage: str = None, **kwargs):
        if status not in VALID_STATUSES:
            raise ValueError(f'Invalid status {status}. Must be one of {VALID_STATUSES}')
        v = self.data['videos'].setdefault(video_id, {'attempts': 0})
        v['status'] = status
        v['error'] = error
        v['stage'] = stage
        v['updated'] = datetime.now(timezone.utc).isoformat()
        v.update(kwargs)
        self.save()

    def set_active_job(self, job_dict: dict | None):
        self.data['active_job'] = job_dict
        self.save()

    def get_status(self, video_id: str) -> str:
        return self.data['videos'].get(video_id, {}).get('status', 'pending')

    def get_counts(self) -> dict:
        return self.data['summary']


def download_with_resume(
    url: str,
    target_mp4: Path,
    expected_md5: str,
    reserve_bytes: int,
    state: AcquisitionState,
    video_id: str,
    stop_event,
) -> bool:
    partial = target_mp4.with_suffix('.part')
    file_size = get_upstream_file_size(url)

    existing_bytes = partial.stat().st_size if partial.is_file() else 0
    if existing_bytes > file_size:
        LOG.warning('Existing .part file (%d bytes) exceeds remote size (%d). Truncating.', existing_bytes, file_size)
        partial.unlink()
        existing_bytes = 0

    needed_bytes = file_size - existing_bytes
    is_safe, free_gb, msg = check_storage_safety(target_mp4.parent, needed_bytes, reserve_bytes)
    if not is_safe:
        raise RuntimeError(msg)

    state.update(video_id, status='downloading', stage='downloading', file_size=file_size)
    state.set_active_job({
        'video_id': video_id,
        'stage': 'downloading',
        'downloaded_bytes': existing_bytes,
        'total_bytes': file_size,
        'percent': round(100 * existing_bytes / max(1, file_size), 1),
    })

    if existing_bytes < file_size:
        curl_cmd = [
            'curl', '-fsSL',
            '--continue-at', '-',
            '--connect-timeout', '20',
            '--speed-time', '45',
            '--speed-limit', '1024',
            '--retry', '4',
            '--retry-delay', '3',
            '--output', str(partial),
            url,
        ]
        start_time = time.monotonic()
        proc = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        while proc.poll() is None:
            if stop_event and stop_event.is_set():
                proc.terminate()
                proc.wait()
                state.set_active_job(None)
                return False

            time.sleep(2)
            cur_bytes = partial.stat().st_size if partial.is_file() else 0
            elapsed = time.monotonic() - start_time
            bytes_delta = cur_bytes - existing_bytes
            speed_kbps = round((bytes_delta / 1024) / max(0.1, elapsed), 1)
            eta_sec = round((file_size - cur_bytes) / max(100, speed_kbps * 1024))
            state.set_active_job({
                'video_id': video_id,
                'stage': 'downloading',
                'downloaded_bytes': cur_bytes,
                'total_bytes': file_size,
                'percent': round(100 * cur_bytes / max(1, file_size), 1),
                'speed_kbps': speed_kbps,
                'eta_seconds': eta_sec,
            })

        if proc.returncode != 0:
            err = proc.stderr.read().decode('utf-8', errors='replace').strip()
            raise RuntimeError(f'curl download failed (exit {proc.returncode}): {err}')

    # Download complete, verify checksum
    state.set_active_job({
        'video_id': video_id,
        'stage': 'verifying_checksum',
        'downloaded_bytes': file_size,
        'total_bytes': file_size,
        'percent': 100.0,
    })
    is_valid, err = verify_media_file(partial, expected_md5=expected_md5, check_opencv=True)
    if not is_valid:
        raise RuntimeError(f'Media verification failed: {err}')

    partial.rename(target_mp4)
    state.update(video_id, status='downloaded', stage='downloaded')
    return True


def run_inference_and_validate(
    mp4_path: Path,
    output_json: Path,
    team_id: str,
    kit,
    state: AcquisitionState,
    video_id: str,
    stop_event,
) -> bool:
    from check_submission import validate_submission

    state.update(video_id, status='processing', stage='inference')
    state.set_active_job({
        'video_id': video_id,
        'stage': 'running_inference',
        'percent': 100.0,
    })

    cmd = [
        sys.executable,
        str(ROOT / 'code/main.py'),
        str(mp4_path),
        str(output_json),
        '--team-id', team_id,
        '--device', 'cpu',
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f'Inference failed: {proc.stderr.strip() or proc.stdout.strip()}')

    if not output_json.is_file():
        raise RuntimeError(f'Expected prediction file {output_json} was not created.')

    result_data = json.loads(output_json.read_text())
    warnings = validate_submission(result_data, kit, filename=output_json, team_id=team_id)

    state.update(video_id, status='validated', stage='completed', warnings=warnings)
    state.set_active_job(None)

    # Immediately delete raw MP4 to preserve disk space!
    if mp4_path.is_file():
        mp4_path.unlink()
        LOG.info('Preserved disk space: cleaned up raw video %s', mp4_path.name)

    return True


def run_pipeline(
    manifest_csv: Path,
    metadata_dir: Path,
    outputs_dir: Path,
    work_dir: Path,
    team_id: str = 'Rajasthani',
    reserve_gb: float = DEFAULT_RESERVE_GIB,
    target_count: int = None,
    only_ids: list[str] = None,
    stop_file: Path = None,
):
    from acquire_manifest import mapped_sources
    from check_submission import StudentKit

    kit = StudentKit(manifest_csv.parent)
    excluded = {'Scuba_KD74Ywuy9ng', 'Scuba_Um4kqB8bQBg'}
    eligible = {k: v for k, v in kit.entries.items() if k not in excluded}

    sources = mapped_sources(kit, metadata_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    state_file = work_dir / 'acquisition_state.json'
    state = AcquisitionState(state_file, team_id=team_id, eligible_keys=list(eligible.keys()))

    # Sync any pre-existing validated outputs in outputs_dir
    for p in outputs_dir.glob('*.json'):
        if p.stem in eligible:
            state.update(p.stem, status='validated', stage='completed')

    counts = state.get_counts()
    print(f"Starting Acquisition Pipeline. Current valid: {counts['validated']} / {counts['total_eligible']}")

    # Identify pending videos sorted by duration
    pending_ids = [
        k for k, v in eligible.items()
        if state.get_status(k) in {'pending', 'failed'}
    ]
    if only_ids:
        pending_ids = [k for k in pending_ids if k in only_ids]

    pending_ids.sort(key=lambda k: (
        not (work_dir / f'{k}.part').is_file(),
        eligible[k].duration_sec,
    ))

    reserve_bytes = int(reserve_gb * GIB)
    class StopEvent:
        def is_set(self):
            return stop_file and stop_file.is_file()

    stop_event = StopEvent()

    for vid in pending_ids:
        if stop_event.is_set():
            print('Stop request detected. Halting pipeline gracefully.')
            break

        if target_count and state.get_counts()['validated'] >= target_count:
            print(f"Target count of {target_count} reached. Stopping.")
            break

        entry = eligible[vid]
        url, md5_hash = sources[vid]
        mp4_target = work_dir / f'{vid}.mp4'
        output_json = outputs_dir / f'{vid}.json'

        print(f"\n[{state.get_counts()['validated']}/{len(eligible)}] Processing {vid} (duration: {entry.duration_sec:.1f}s)...")
        try:
            download_with_resume(
                url=url,
                target_mp4=mp4_target,
                expected_md5=md5_hash,
                reserve_bytes=reserve_bytes,
                state=state,
                video_id=vid,
                stop_event=stop_event,
            )

            run_inference_and_validate(
                mp4_path=mp4_target,
                output_json=output_json,
                team_id=team_id,
                kit=kit,
                state=state,
                video_id=vid,
                stop_event=stop_event,
            )
            print(f"Successfully processed and validated {vid}!")

        except Exception as exc:
            print(f"Error on {vid}: {exc}")
            state.update(vid, status='failed', error=str(exc))
            state.set_active_job(None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'reference/student_package/manifest.csv')
    parser.add_argument('--metadata', type=Path, default=ROOT.parent / 'runtime/track4-assets/full-manifest')
    parser.add_argument('--outputs', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--work', type=Path, default=ROOT.parent / 'runtime/track4-assets/full-manifest/Full Run')
    parser.add_argument('--team-id', default='Rajasthani')
    parser.add_argument('--reserve-gb', type=float, default=DEFAULT_RESERVE_GIB)
    parser.add_argument('--target-count', type=int, help='Stop after this many manifest outputs pass local validation')
    parser.add_argument('--only', nargs='+', help='Run only these specific video IDs')
    parser.add_argument('--stop-file', type=Path, help='File whose existence signals graceful shutdown')
    args = parser.parse_args()

    run_pipeline(
        manifest_csv=args.manifest,
        metadata_dir=args.metadata,
        outputs_dir=args.outputs,
        work_dir=args.work,
        team_id=args.team_id,
        reserve_gb=args.reserve_gb,
        target_count=args.target_count,
        only_ids=args.only,
        stop_file=args.stop_file,
    )


if __name__ == '__main__':
    main()
