"""Acquire exact manifest sources and run inference with bounded disk usage."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from hashlib import file_digest
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from threading import Condition, Event, Lock

from run_manifest import ROOT, StudentKit, atomic_json, inspect_outputs

EPIC_55 = 'https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d'
EPIC_100 = 'https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m'
GIB = 1024 ** 3
FREE_RESERVE = 3 * GIB
FPV_MAX = GIB // 2
CURL = ['curl', '-fsSL', '--connect-timeout', '15']


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def mapped_sources(kit, metadata):
    splits = {row['video_id']: row['split']
              for row in rows(metadata / 'epic_55_splits.csv')}
    checksums = {row['file_remote_path']: row['md5']
                 for row in rows(metadata / 'md5.csv')}
    errata = {row['rdsf_path']: row['dropbox_path']
              for row in rows(metadata / 'errata.csv')}
    fpv = {row['video_name']: row['youtube_id']
           for row in rows(metadata / 'fpvsum-videolist.csv')}
    result = {}
    for video_id, entry in kit.entries.items():
        if entry.dataset == 'fpvsum':
            youtube_id = fpv.get(video_id, '')
            if not re.fullmatch(r'[A-Za-z0-9_-]{11}', youtube_id):
                raise ValueError(f'No valid official FPVSum mapping for {video_id}')
            result[video_id] = (f'https://www.youtube.com/watch?v={youtube_id}', None)
        else:
            if not re.fullmatch(r'P[0-9]{2}_[0-9]{2,3}', video_id):
                raise ValueError(f'Invalid EPIC ID {video_id}')
            participant, part = video_id.split('_')
            if len(part) == 3:
                path, base = f'{participant}/videos/{video_id}.MP4', EPIC_100
            else:
                path = f'videos/{splits[video_id]}/{participant}/{video_id}.MP4'
                base = EPIC_55
            result[video_id] = (errata.get(path, f'{base}/{path}'), checksums[path])
    return result


def command(args, timeout=None):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
        raise RuntimeError(lines[-1] if lines else f'Command returned {result.returncode}')
    return result.stdout


def source_size(url):
    response = command([*CURL, '--head', '--max-time', '30', url])
    lengths = re.findall(r'^Content-Length: (\d+)\s*$', response, re.I | re.M)
    if not lengths:
        raise RuntimeError('Upstream supplied no file size')
    return int(lengths[-1])


def epic_download(path, partial, url, expected_md5, file_size=None):
    if not path.exists():
        for attempt in range(3):
            try:
                if not (partial.is_file() and partial.stat().st_size == file_size):
                    command([*CURL, '--continue-at', '-', '--speed-time', '60',
                             '--speed-limit', '1024', '--max-time', '7200',
                             '--output', str(partial), url])
                if partial.stat().st_size != file_size:
                    raise RuntimeError('Downloaded size differs from the official source')
                partial.rename(path)
                break
            except RuntimeError:
                if attempt == 2:
                    raise
    with path.open('rb') as stream:
        if file_digest(stream, 'md5').hexdigest() != expected_md5:
            raise RuntimeError('Official EPIC checksum mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kit', type=Path)
    parser.add_argument('metadata', type=Path)
    parser.add_argument('outputs', type=Path)
    parser.add_argument('work', type=Path, help='Dedicated directory for managed downloads and progress')
    parser.add_argument('--team-id', required=True)
    parser.add_argument('--node', help='Optional Node executable for yt-dlp')
    parser.add_argument('--seed-dir', type=Path, help='Previously acquired files to preserve')
    parser.add_argument('--defer', nargs='*', default=(), help='IDs currently downloading in another process')
    parser.add_argument('--only', nargs='+', help='Run a selected set of manifest IDs')
    parser.add_argument('--epic-workers', type=int, choices=range(1, 9), default=3)
    parser.add_argument('--target-count', type=int, help='Stop after this many manifest outputs pass local validation')
    parser.add_argument('--fpv-workers', type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    kit = StudentKit(args.kit)
    if args.target_count is not None and not 1 <= args.target_count <= len(kit.entries):
        parser.error('Target count must be between one and the manifest size')
    sources = mapped_sources(kit, args.metadata)
    args.work.mkdir(parents=True, exist_ok=True)
    args.outputs.mkdir(parents=True, exist_ok=True)
    report = args.work / 'acquisition.json'
    history = {}
    if report.is_file():
        previous = json.loads(report.read_text())
        if previous['team_id'] == args.team_id:
            history = previous['videos']
    valid, invalid, duplicate = inspect_outputs(args.outputs, kit, args.team_id)
    if invalid or duplicate:
        parser.error('Resolve invalid or duplicate output files before resuming')
    ids = set(args.only or kit.entries) - set(args.defer)
    if ids - kit.entries.keys():
        parser.error('Requested video ID is absent from the manifest')
    pending = sorted((entry for key, entry in kit.entries.items() if key in ids and key not in valid),
                     key=lambda entry: (not (args.work / (entry.video_id + '.part')).exists(),
                                        entry.duration_sec))
    state_lock = Lock()
    disk_budget = Condition(state_lock)
    infer_lock = Lock()
    stop = Event()
    count = len(valid)
    if args.target_count and count >= args.target_count:
        return 0
    reserved = 0
    ledger = {
        key: {'dataset': entry.dataset, 'duration_sec': entry.duration_sec,
              'source_url': sources[key][0],
              'status': 'valid_output' if key in valid else 'queued',
              'attempts': history.get(key, {}).get('attempts', 0)}
        for key, entry in kit.entries.items()
    }

    def record(key=None, status=None, error=None, stage=None):
        nonlocal count
        with disk_budget:
            if key is not None:
                ledger[key].update(status=status, error=error, stage=stage)
            count = sum(row['status'] == 'valid_output' for row in ledger.values())
            if args.target_count and count >= args.target_count:
                stop.set()
                disk_budget.notify_all()
            atomic_json(report, {'team_id': args.team_id, 'expected': len(ledger),
                                 'target': args.target_count,
                                 'valid_outputs': count, 'videos': ledger})
            if key:
                print(f'{count}/{len(ledger)} {key} {status}'
                      f'{": " + error if error else ""}', flush=True)

    def process(entry):
        nonlocal reserved
        if stop.is_set():
            return
        key = entry.video_id
        url, expected_md5 = sources[key]
        stage = 'acquisition'
        budget = 0
        source_size_bytes = None
        managed = args.work / (key + '.mp4')
        partial = args.work / (key + '.part')
        seed = args.seed_dir / managed.name if args.seed_dir else None
        try:
            with state_lock:
                ledger[key]['attempts'] += 1
            record(key, 'acquiring', stage=stage)
            if seed and seed.is_file():
                source = seed
                if expected_md5:
                    epic_download(source, partial, url, expected_md5)
            else:
                source = managed
                need = 0
                if not source.exists():
                    if expected_md5:
                        try:
                            source_size_bytes = source_size(url)
                        except RuntimeError:
                            alternative = url.replace('https://data.bris.ac.uk/', 'https://data.bristol.ac.uk/')
                            if alternative == url:
                                raise
                            url = alternative
                            source_size_bytes = source_size(url)
                        previous_bytes = partial.stat().st_size if partial.is_file() else 0
                        if previous_bytes > source_size_bytes:
                            raise RuntimeError('Partial file exceeds expected source size')
                        need = source_size_bytes - previous_bytes
                    else:
                        need = FPV_MAX
                with disk_budget:
                    while not stop.is_set() and need > shutil.disk_usage(args.work).free - reserved - FREE_RESERVE:
                        if not reserved:
                            raise RuntimeError('Insufficient space for this download with disk reserve')
                        disk_budget.wait()
                    if stop.is_set():
                        return
                    reserved += need
                    budget = need
                if expected_md5:
                    epic_download(source, partial, url, expected_md5, source_size_bytes)
                elif not source.exists():
                    flags = ['--js-runtimes', f'node:{args.node}'] if args.node else []
                    command([sys.executable, '-m', 'yt_dlp', '--no-playlist', '--no-progress',
                             '--socket-timeout', '15', '--retries', '2', '--max-filesize', '512M',
                             '-f', 'bv[ext=mp4][height<=720]/b[ext=mp4][height<=720]',
                             '-o', str(source), *flags, url], timeout=1800)
                    if not source.is_file():
                        raise RuntimeError('No compatible MP4 was downloaded')
            stage = 'inference'
            record(key, 'downloaded', stage=stage)
            output = args.outputs / (key + '.json')
            with infer_lock:
                if stop.is_set():
                    return
                record(key, 'processing', stage=stage)
                command([sys.executable, str(ROOT / 'code/main.py'), str(source), str(output),
                         '--team-id', args.team_id, '--device', 'cpu'])
                checked, errors, _ = inspect_outputs(args.outputs, kit, args.team_id)
                if key not in checked:
                    raise RuntimeError(errors.get(output.name, 'Output failed local validation'))
                if source == managed:
                    managed.unlink()
                record(key, 'valid_output')
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            record(key, 'failed', str(exc), stage)
        finally:
            with disk_budget:
                reserved -= budget
                disk_budget.notify_all()

    record()
    with ThreadPoolExecutor(args.epic_workers) as epic, ThreadPoolExecutor(args.fpv_workers) as fpv:
        futures = [(fpv if entry.dataset == 'fpvsum' else epic).submit(process, entry)
                   for entry in pending]
        for future in as_completed(futures):
            future.result()
    record()
    if args.target_count:
        return int(count < args.target_count)
    return int(any(ledger[key]['status'] != 'valid_output' for key in ids))


if __name__ == '__main__':
    raise SystemExit(main())
