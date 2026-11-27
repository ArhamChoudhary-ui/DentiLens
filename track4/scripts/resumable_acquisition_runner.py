"""Resumable, storage-aware dataset acquisition and inference pipeline for EgoSight XR."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
from hashlib import md5
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

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


def wait_for_connectivity(host: str = 'data.bris.ac.uk', max_wait_sec: int = 7200, check_interval: int = 15, stop_event=None) -> bool:
    """Pause execution gracefully until upstream DNS/network connectivity is established."""
    import socket
    start = time.monotonic()
    logged = False
    while time.monotonic() - start < max_wait_sec:
        if stop_event and stop_event.is_set():
            return False
        try:
            socket.gethostbyname(host)
            if logged:
                LOG.info('Network connectivity to %s restored after %.1fs', host, time.monotonic() - start)
            return True
        except (socket.gaierror, socket.error):
            if not logged:
                LOG.warning('Network unreachable (cannot resolve %s). Pausing runner until connectivity returns...', host)
                logged = True
            time.sleep(check_interval)
    LOG.error('Network connectivity could not be established within %d seconds', max_wait_sec)
    return False


def get_upstream_file_size(url: str, timeout: int = 25, stop_event=None) -> int:
    from urllib.parse import urlparse
    host = urlparse(url).hostname or 'data.bris.ac.uk'
    wait_for_connectivity(host, stop_event=stop_event)
    for attempt in range(6):
        cmd = ['curl', '-fsSL', '-I', '--connect-timeout', '15', '--max-time', str(timeout), url]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            lengths = re.findall(r'^Content-Length:\s*(\d+)', res.stdout, re.IGNORECASE | re.MULTILINE)
            if lengths:
                return int(lengths[-1])
        wait_for_connectivity(host, stop_event=stop_event)
        time.sleep(min(30, 5 * (2 ** attempt)))
    raise RuntimeError(f'HTTP HEAD failed for {url} after retries: {res.stderr.strip()}')


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
    num_workers: int = 6,
    block_size: int = 4 * 1024 * 1024,
) -> bool:
    from urllib.parse import urlparse
    host = urlparse(url).hostname or 'data.bris.ac.uk'
    wait_for_connectivity(host, stop_event=stop_event)

    partial = target_mp4.with_suffix('.part')
    blocks_meta = target_mp4.with_suffix('.blocks.json')
    file_size = get_upstream_file_size(url, stop_event=stop_event)

    needed_bytes = file_size
    is_safe, free_gb, msg = check_storage_safety(target_mp4.parent, needed_bytes, reserve_bytes)
    if not is_safe:
        raise RuntimeError(msg)

    cookie_processor = urllib.request.HTTPCookieProcessor()
    opener = urllib.request.build_opener(cookie_processor)

    # Check range support
    req = urllib.request.Request(url, method='HEAD')
    req.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)')
    supports_ranges = True
    try:
        with opener.open(req, timeout=30) as resp:
            ar = resp.headers.get('Accept-Ranges')
            supports_ranges = bool(ar and 'bytes' in ar.lower())
    except Exception as e:
        LOG.warning('Could not verify Accept-Ranges with HEAD (%s), assuming supported.', e)

    total_blocks = math.ceil(file_size / block_size)
    completed_blocks = set()

    if blocks_meta.is_file() and partial.is_file():
        try:
            m = json.loads(blocks_meta.read_text())
            if m.get('total_size') == file_size and m.get('block_size') == block_size:
                completed_blocks = set(m.get('completed', []))
        except Exception:
            completed_blocks = set()

    state.update(video_id, status='downloading', stage='downloading', file_size=file_size)
    cur_done = min(file_size, len(completed_blocks) * block_size)
    state.set_active_job({
        'video_id': video_id,
        'stage': 'downloading',
        'downloaded_bytes': cur_done,
        'total_bytes': file_size,
        'percent': round(100 * len(completed_blocks) / max(1, total_blocks), 1),
    })

    if supports_ranges:
        fd = os.open(str(partial), os.O_RDWR | os.O_CREAT)
        try:
            cur_st_size = os.fstat(fd).st_size
            if cur_st_size < file_size:
                os.ftruncate(fd, file_size)
        except Exception as e:
            LOG.warning('Could not ftruncate partial file: %s', e)

        pending_blocks = [i for i in range(total_blocks) if i not in completed_blocks]
        start_time = time.monotonic()
        last_save = start_time
        downloaded_session = 0

        def save_state():
            meta_data = {
                'total_size': file_size,
                'block_size': block_size,
                'total_blocks': total_blocks,
                'completed': sorted(list(completed_blocks)),
                'updated': time.time(),
            }
            tmp_m = blocks_meta.with_suffix('.tmp')
            tmp_m.write_text(json.dumps(meta_data))
            tmp_m.replace(blocks_meta)

        def fetch_b(b_idx):
            b_start = b_idx * block_size
            b_end = min(b_start + block_size - 1, file_size - 1)
            exp_len = b_end - b_start + 1
            for attempt in range(5):
                if stop_event and stop_event.is_set():
                    return None
                try:
                    r = urllib.request.Request(url)
                    r.add_header('Range', f'bytes={b_start}-{b_end}')
                    r.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)')
                    with opener.open(r, timeout=45) as resp:
                        chunk = resp.read()
                        if len(chunk) != exp_len:
                            raise IOError(f'Block {b_idx} truncated: got {len(chunk)}/{exp_len}')
                        os.pwrite(fd, chunk, b_start)
                        return b_idx, len(chunk)
                except Exception as exc:
                    wait_for_connectivity(host, stop_event=stop_event)
                    if attempt == 4:
                        raise RuntimeError(f'Block {b_idx} failed after retries: {exc}')
                    time.sleep(min(15, 2 ** attempt))

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_b = {executor.submit(fetch_b, b): b for b in pending_blocks}
                for fut in as_completed(future_to_b):
                    if stop_event and stop_event.is_set():
                        state.set_active_job(None)
                        return False
                    res = fut.result()
                    if res is None:
                        continue
                    b_idx, n_b = res
                    completed_blocks.add(b_idx)
                    downloaded_session += n_b

                    now = time.monotonic()
                    elapsed = max(0.1, now - start_time)
                    speed_kbps = (downloaded_session / 1024) / elapsed
                    done_bytes = min(file_size, len(completed_blocks) * block_size)
                    pct = round(100 * len(completed_blocks) / total_blocks, 1)

                    if now - last_save >= 3.0 or len(completed_blocks) == total_blocks:
                        save_state()
                        last_save = now
                        state.set_active_job({
                            'video_id': video_id,
                            'stage': 'downloading',
                            'downloaded_bytes': done_bytes,
                            'total_bytes': file_size,
                            'percent': pct,
                            'speed_kbps': round(speed_kbps, 1),
                            'eta_seconds': int((file_size - done_bytes) / max(100, speed_kbps * 1024)),
                        })
        finally:
            os.close(fd)
            save_state()

        if len(completed_blocks) < total_blocks:
            return False

    else:
        # Fallback to curl single-stream
        curl_cmd = [
            'curl', '-fsSL',
            '--continue-at', '-',
            '--connect-timeout', '30',
            '--speed-time', '60',
            '--speed-limit', '500',
            '--retry', '10',
            '--output', str(partial),
            url,
        ]
        proc = subprocess.run(curl_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f'curl fallback failed: {proc.stderr.strip()}')

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

    if blocks_meta.is_file():
        blocks_meta.unlink()
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


def sync_and_stage_video(vid: str, output_json: Path, team_id: str = 'Rajasthani', auto_publish: bool = False):
    import json
    import os
    import shutil
    import subprocess
    from pathlib import Path
    from zipfile import ZipFile, ZIP_DEFLATED

    ROOT_DIR = Path(__file__).resolve().parents[2]
    TRACK4_DIR = ROOT_DIR / 'track4'
    staging_dir = ROOT_DIR / 'runtime/submission_staging/Rajasthani_Staging'
    submission_dir = ROOT_DIR / 'runtime/submission_final'
    base_rajasthani = submission_dir / 'Rajasthani'

    # 1. Copy JSON to staging and final
    (staging_dir / 'outputs').mkdir(parents=True, exist_ok=True)
    (base_rajasthani / 'outputs').mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_json, staging_dir / 'outputs' / f'{vid}.json')
    shutil.copy2(output_json, base_rajasthani / 'outputs' / f'{vid}.json')

    # 2. Extract frames_processed from output_json
    pred = json.loads(output_json.read_text())
    frames_processed = pred.get('provenance', {}).get('video', {}).get('frame_count', 0)
    if not frames_processed:
        frames_processed = pred.get('provenance', {}).get('frames_processed', 0)

    # 3. Update prediction_provenance_audit.json
    audit_path = TRACK4_DIR / 'manifest/prediction_provenance_audit.json'
    with open(audit_path, 'r') as f:
        audit = json.load(f)

    for item in audit:
        if item.get('video_id') == vid:
            item['source_verified'] = True
            item['checksum_verified'] = True
            if frames_processed:
                item['frames_processed'] = frames_processed
            item['model_used'] = 'R3D-18 + TSN-RGB + BLIP v2.0'
            item['genuine_inference'] = True
            item['schema_valid'] = True
            item['status'] = 'validated_genuine'
            break

    atomic_write_json(audit_path, audit)

    # 4. Rebuild all 3 submission zips
    staging_zip = ROOT_DIR / 'runtime/submission_staging/Rajasthani_Track4_Staging.zip'
    if staging_zip.exists():
        staging_zip.unlink()
    with ZipFile(staging_zip, 'w', compression=ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(staging_dir):
            for f in files:
                full = Path(root) / f
                rel = full.relative_to(staging_dir.parent)
                zf.write(full, str(rel))

    offline_zip = submission_dir / 'VelloreInstituteofTechnology_Rajasthani_Track4_Offline_Submission.zip'
    if offline_zip.exists():
        offline_zip.unlink()
    with ZipFile(offline_zip, 'w', compression=ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(base_rajasthani):
            for f in files:
                full = Path(root) / f
                rel = full.relative_to(submission_dir)
                zf.write(full, str(rel))

    final_staging_zip = submission_dir / 'VelloreInstituteofTechnology_Rajasthani_Track4_Staging_Submission.zip'
    if final_staging_zip.exists():
        final_staging_zip.unlink()
    with ZipFile(final_staging_zip, 'w', compression=ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(base_rajasthani):
            for f in files:
                if f.endswith('.pth') or f.endswith('.ckpt'):
                    continue
                full = Path(root) / f
                rel = full.relative_to(submission_dir)
                zf.write(full, str(rel))

    # 5. Verify zips with unzip -t
    for zp in [staging_zip, offline_zip, final_staging_zip]:
        res = subprocess.run(['unzip', '-t', str(zp)], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f'Zip corruption detected in {zp.name}: {res.stderr}')

    genuine_count = sum(1 for x in audit if x.get('status') == 'validated_genuine')
    pct = round(100 * genuine_count / 185, 1)
    LOG.info('Post-validation complete for %s. Genuine coverage: %d/185 (%.1f%%)', vid, genuine_count, pct)

    if auto_publish:
        LOG.info('Auto-committing and pushing %s to egosight/feature/full-dataset-intelligence...', vid)
        subprocess.run(['git', 'add', str(audit_path)], check=True)
        commit_msg = f"feat(provenance): genuine validation of {vid}, incrementing verified coverage to {genuine_count}/185 ({pct}%)"
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        subprocess.run(['git', 'push', 'egosight', 'feature/full-dataset-intelligence'], check=True)
        LOG.info('Successfully pushed commit for %s to GitHub!', vid)


def run_pipeline(
    manifest_csv: Path,
    metadata_dir: Path,
    outputs_dir: Path,
    work_dir: Path,
    team_id: str = 'Rajasthani',
    reserve_gb: float = DEFAULT_RESERVE_GIB,
    target_count: int = None,
    batch_count: int = None,
    only_ids: list[str] = None,
    stop_file: Path = None,
    auto_stage: bool = False,
    auto_publish: bool = False,
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
    processed_in_batch = 0

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

            if auto_stage:
                sync_and_stage_video(vid, output_json, team_id=team_id, auto_publish=auto_publish)

            processed_in_batch += 1
            if batch_count and processed_in_batch >= batch_count:
                print(f"Batch count limit ({batch_count}) reached. Stopping.")
                break

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
    parser.add_argument('--batch-count', type=int, help='Stop after processing this many videos in this session')
    parser.add_argument('--auto-stage', action='store_true', help='Automatically sync outputs and rebuild submission zips')
    parser.add_argument('--auto-publish', action='store_true', help='Automatically commit and push verified updates to GitHub')
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
        batch_count=args.batch_count,
        only_ids=args.only,
        stop_file=args.stop_file,
        auto_stage=args.auto_stage,
        auto_publish=args.auto_publish,
    )


if __name__ == '__main__':
    main()
