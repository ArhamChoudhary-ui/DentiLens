"""Dental-photo model inference on source-timestamped video frames."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np


def pick_highlights(observations, duration, clip_seconds=4.0):
    """Rank real observation windows by confidence and new model classes."""
    budget = duration * 0.15
    if budget < clip_seconds:
        return []
    candidates = {}
    for observation in observations:
        start = min(math.floor(observation['timestamp_sec'] / clip_seconds) * clip_seconds,
                    max(0.0, duration - clip_seconds))
        candidates.setdefault(start, []).append(observation)
    chosen, seen = [], set()
    while candidates and budget >= clip_seconds - 1e-8:
        def utility(start):
            items = candidates[start]
            classes = {item['model_label'] for item in items}
            confidence = float(np.mean([item['confidence'] for item in items]))
            novelty = len(classes - seen) / max(1, len(classes))
            return 0.7 * confidence + 0.3 * novelty
        ranked = sorted(candidates, key=lambda start: (-utility(start), start))
        start = next((start for start in ranked if all(
            start + clip_seconds <= clip['start_sec'] or start >= clip['end_sec']
            for clip in chosen)), None)
        if start is None:
            break
        importance = utility(start)
        items = candidates.pop(start)
        chosen.append({'start_sec': start, 'end_sec': start + clip_seconds,
                       'importance': importance,
                       'reason': 'Model-supported visual observations',
                       'model_labels': sorted({item['model_label'] for item in items})})
        seen.update(item['model_label'] for item in items)
        budget -= clip_seconds
    return sorted(chosen, key=lambda clip: clip['start_sec'])


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def export_clips(video, destination, highlights):
    import imageio_ffmpeg
    filters = []
    for index, clip in enumerate(highlights):
        filters.append(f"[0:v]trim=start={clip['start_sec']}:end={clip['end_sec']},setpts=PTS-STARTPTS[v{index}]")
    inputs = ''.join(f'[v{index}]' for index in range(len(highlights)))
    filters.append(inputs + f'concat=n={len(highlights)}:v=1:a=0[out]')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-nostdin', '-y',
                    '-i', str(video), '-filter_complex', ';'.join(filters), '-map', '[out]',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    str(destination)], capture_output=True, check=True, timeout=600)


def analyze(video, weights, output, sample_fps=1.0, confidence=0.35, device='cpu', predictor=None):
    video, weights, output = Path(video).resolve(), Path(weights).resolve(), Path(output).resolve()
    if not video.is_file() or video.suffix.lower() != '.mp4':
        raise ValueError('Provide an existing MP4 video')
    if predictor is None and not weights.is_file():
        raise ValueError('The trained dental checkpoint is missing')
    if output.exists():
        raise ValueError('Choose a new output directory for this analysis')
    if not math.isfinite(sample_fps) or not 0 < sample_fps <= 5:
        raise ValueError('Sampling rate must be in (0, 5] frames per second')
    if not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError('Confidence threshold must be in (0, 1]')
    with video.open('rb') as stream:
        input_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    expected_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or not math.isfinite(expected_frames) or expected_frames <= 0:
        cap.release()
        raise ValueError('Video has unreadable or invalid timing metadata')
    model = None
    if predictor is None:
        from ultralytics import YOLO
        try:
            model = YOLO(str(weights))
            if model.task != 'segment':
                raise ValueError('Expected the AlphaDent segmentation checkpoint')
        except Exception:
            cap.release()
            raise
    step = max(1, round(fps / sample_fps))
    output.parent.mkdir(parents=True, exist_ok=True)
    observations, sampled, decoded = [], 0, 0
    try:
        with tempfile.TemporaryDirectory(prefix='.dental-', dir=output.parent) as staging:
            stage = Path(staging)
            evidence = stage / 'evidence'
            evidence.mkdir()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                index = decoded
                decoded += 1
                if index % step:
                    continue
                sampled += 1
                timestamp = index / fps
                frame_observations = []
                if predictor is not None:
                    detections, annotated = predictor.predict(frame)
                    for detection_index, detection in enumerate(detections):
                        mask = detection.pop('mask')
                        mask_path = f'evidence/frame-{index:08d}-mask-{detection_index:02d}.png'
                        if not cv2.imwrite(str(stage / mask_path), mask * 255):
                            raise ValueError('Could not save predicted region mask')
                        frame_observations.append({**detection, 'timestamp_sec': timestamp,
                            'source_frame': index, 'mask_path': mask_path,
                            'mask_size': [int(mask.shape[1]), int(mask.shape[0])],
                            'source_size': [int(frame.shape[1]), int(frame.shape[0])],
                            'status': 'Unverified model observation requiring professional review',
                            'evidence': f'evidence/frame-{index:08d}.jpg'})
                else:
                    prediction = model.predict(frame, conf=confidence, imgsz=640, device=device,
                                               verbose=False, save=False)[0]
                    h, w = frame.shape[:2]
                    for box in prediction.boxes:
                        class_id = int(box.cls.item())
                        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                        normalized = [max(0.0, min(1.0, x1 / w)), max(0.0, min(1.0, y1 / h)),
                                      max(0.0, min(1.0, x2 / w)), max(0.0, min(1.0, y2 / h))]
                        if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
                            continue
                        frame_observations.append({
                            'timestamp_sec': timestamp, 'source_frame': index,
                            'model_class_id': class_id, 'model_label': model.names[class_id],
                            'confidence': float(box.conf.item()), 'bbox_xyxy_normalized': normalized,
                            'status': 'Unverified model observation requiring professional review',
                            'evidence': f'evidence/frame-{index:08d}.jpg'})
                    annotated = prediction.plot() if frame_observations else None
                if frame_observations:
                    # Only annotate model-predicted source frames; the original video stays available.
                    if not cv2.imwrite(str(evidence / f'frame-{index:08d}.jpg'), annotated):
                        raise ValueError('Could not save source-frame evidence')
                    observations.extend(frame_observations)
            if decoded != int(expected_frames):
                raise ValueError(f'Incomplete video decode: {decoded}/{int(expected_frames)} frames')
            duration = decoded / fps
            highlights = pick_highlights(observations, duration)
            if predictor is None:
                with weights.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                model_description = {'name': 'AlphaDent dental-photo segmentation',
                    'weights_sha256': digest, 'classes': model.names,
                    'confidence_threshold': confidence, 'device': device}
            else:
                model_description = {'name': getattr(predictor, 'model_name', 'Grounding DINO + SAM 2.1 tiny'),
                    'provenance': predictor.provenance, 'classes': predictor.names,
                    'confidence_threshold': confidence, 'device': device}
                if predictor.prompt:
                    model_description['text_prompt'] = predictor.prompt
                else:
                    model_description['localization_method'] = 'Trained intraoral tooth detector boxes'
            report = {
                'schema_version': 'dentilens-dental-1', 'video': video.name,
                'input_sha256': input_digest,
                'duration_sec': duration, 'fps': fps, 'frames_decoded': decoded,
                'sampled_frames': sampled, 'effective_sample_fps': fps / step,
                'model': model_description,
                'observations': observations, 'highlights': highlights,
                'highlight_duration_sec': sum(clip['end_sec']-clip['start_sec'] for clip in highlights),
                'capabilities': {'visual_observations': True, 'procedure_action_recognition': False,
                                 'individual_tooth_numbering': False},
                'limitations': ['Model predictions on dental video require evaluation against annotations.',
                                'CFR video timing required. Fast events may fall between sampled frames.',
                                'Observation counts include repeated detections across frames.',
                                'A teeth mask may cover several adjacent teeth; individual numbering is unavailable.',
                                'Masks are independently predicted per sampled frame; temporal tracking is unavailable.',
                                'Highlight video uses original frames and omits audio.'] + getattr(predictor, 'limitations', []),
                'review_notice': 'Research prototype. All model observations require professional review.'}
            if highlights:
                export_clips(video, stage / 'highlights.mp4', highlights)
            write_json(stage / 'report.json', report)
            stage.rename(output)
            return report
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description='Analyze dental video with photo-based or text-grounded segmentation')
    parser.add_argument('video', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--weights', type=Path, default=Path(__file__).resolve().parents[1] / 'models/alphadent.pt')
    parser.add_argument('--sample-fps', type=float, default=1.0)
    parser.add_argument('--confidence', type=float,
                        help='Defaults to 0.50 for text-guided regions or 0.35 for AlphaDent and the experimental tooth detector')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--engine', choices=['alphadent', 'grounded-sam2', 'teeth-mirror', 'tooth-sam2', 'tooth-types'], default='grounded-sam2')
    parser.add_argument('--tooth-weights', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'models/tooth-detector.pt')
    parser.add_argument('--model-directory', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'models/grounded')
    args = parser.parse_args()
    if args.confidence is None:
        args.confidence = 0.5 if args.engine in ('grounded-sam2', 'teeth-mirror') else 0.35
    try:
        predictor = None
        if args.engine in ('grounded-sam2', 'teeth-mirror'):
            from grounded_sam import GroundedSegmenter
            predictor = GroundedSegmenter(args.model_directory, args.confidence, args.device, include_mirror=args.engine == 'teeth-mirror')
        elif args.engine in ('tooth-sam2', 'tooth-types'):
            from tooth_sam import ToothSegmenter
            typed = args.engine == 'tooth-types'
            weights = args.tooth_weights
            if typed and weights.name == 'tooth-detector.pt':
                weights = weights.with_name('tooth-types.pt')
            predictor = ToothSegmenter(weights, args.model_directory, args.confidence, args.device, tooth_types=typed)
        report = analyze(args.video, args.weights, args.output, args.sample_fps,
                         args.confidence, args.device, predictor=predictor)
    except Exception as exc:
        parser.exit(1, f'Analysis failed: {exc}\n')
    print(f"Analyzed {report['sampled_frames']} frames; {len(report['observations'])} visual observations")
    print(f"Report saved to {args.output / 'report.json'}")


if __name__ == '__main__':
    main()
