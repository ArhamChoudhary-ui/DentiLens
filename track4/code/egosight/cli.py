import argparse
import logging
from time import perf_counter
from pathlib import Path
import sys

from .pipeline import analyze
from .schema import atomic_json
from .contract import StudentKit, configured_team_id, to_submission
from .video import VideoReader


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description='EgoSight XR: real action inference or explicitly labeled development modes')
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--team-id', default=configured_team_id())
    p.add_argument('--format', choices=['submission', 'review'], default='submission')
    p.add_argument('--development-video', action='store_true',
                   help='Allow an ID outside the official manifest for explicit development runs')
    p.add_argument('--mode', choices=['model','motion-baseline','infrastructure'], default='model')
    p.add_argument('--checkpoint', type=Path, help='Optional custom VideoMAE heads')
    p.add_argument('--weights-root', type=Path, default=root/'weights')
    p.add_argument('--dataset', choices=['epic_kitchens_100','fpvsum'],
                   help='Select the research head for a non-manifest development video')
    local_encoder = root/'weights/encoder'
    p.add_argument('--encoder-path', default=str(local_encoder) if local_encoder.is_dir() else None,
                   help='Local pretrained encoder directory for offline submission')
    p.add_argument('--device', choices=['auto','cpu','cuda','mps'], default='auto')
    p.add_argument('--allow-download', action='store_true', help='Allow encoder download; offline by default')
    p.add_argument('--allow-short', action='store_true', help='Compatibility option; manifest videos of all lengths are accepted')
    p.add_argument('--strategy', choices=['diversity','uniform','motion','confidence'], default='diversity')
    p.add_argument('--report', type=Path, help='Development provenance, outside the submission outputs directory')
    p.add_argument('--highlight-mp4', type=Path)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        paths = [args.input.resolve(), args.output.resolve()]
        paths += [x.resolve() for x in (args.report, args.highlight_mp4) if x is not None]
        if len(paths) != len(set(paths)):
            raise ValueError('Input, JSON, report and MP4 paths must be distinct')
        kit = StudentKit()
        entry = kit.entries.get(args.input.stem)
        if entry and args.dataset and entry.dataset!=args.dataset:
            raise ValueError('Dataset differs from the official manifest')
        dataset=entry.dataset if entry else args.dataset
        if args.format == 'submission':
            if not args.team_id or not args.team_id.strip():
                raise ValueError('Set your team ID in team_id.txt, EGOSIGHT_TEAM_ID or --team-id')
            if args.output.name != args.input.stem + '.json':
                raise ValueError('Submission output must be named <input_video_id>.json')
            if not args.development_video and args.input.stem not in kit.entries:
                raise ValueError('Input video ID is absent from the official manifest')
            if not args.development_video:
                decoded_duration = VideoReader(args.input).info.duration
                expected_duration = kit.entries[args.input.stem].duration_sec
                if abs(decoded_duration - expected_duration) > 0.5 + 1e-9:
                    raise ValueError('Input duration differs from the manifest by more than 0.5s')
        started=perf_counter()
        result, provenance = analyze(args.input, args.mode, args.checkpoint, args.device,
                                     not args.allow_download, args.allow_short, args.strategy, args.encoder_path, dataset, args.weights_root)
        provenance['analysis_seconds']=perf_counter()-started
        if args.format == 'submission':
            result, warnings = to_submission(result, provenance, args.team_id, kit,
                                            not args.development_video, args.output)
            provenance['contract_warnings'] = warnings
            provenance['schema_source'] = 'Supplied TEAM_README.md and supported_classes.json'
            for warning in warnings:
                logging.warning('%s', warning)
        if args.highlight_mp4:
            from .export import export_highlights
            export_highlights(args.input, args.highlight_mp4,
                              [{'start': h['segment'][0], 'end': h['segment'][1]} for h in result['highlights']]
                              if args.format == 'submission' else result['highlights'])
        atomic_json(args.output, result)
        if args.report:
            atomic_json(args.report, provenance)
        logging.info('Wrote %s (%s); organizer validator has not been run', args.output, args.mode)
        if args.mode != 'model':
            logging.warning('DEVELOPMENT OUTPUT ONLY: no semantic predictions; not submission-ready')
        return 0
    except Exception as exc:
        logging.error('%s', exc)
        return 1
