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


def analyze(video, weights, output, sample_fps=1.0, confidence=0.35, device='cpu'):
    from ultralytics import YOLO
    video, weights, output = Path(video).resolve(), Path(weights).resolve(), Path(output).resolve()
    if not video.is_file() or video.suffix.lower() != '.mp4':
        raise ValueError('Provide an existing MP4 video')
    if not weights.is_file():
        raise ValueError('The trained dental checkpoint is missing')
    if output.exists():
        raise ValueError('Choose a new output directory for this analysis')
    if not math.isfinite(sample_fps) or not 0 < sample_fps <= 5:
        raise ValueError('Sampling rate must be in (0, 5] frames per second')
    if not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError('Confidence threshold must be in (0, 1]')
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    expected_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or not math.isfinite(expected_frames) or expected_frames <= 0:
        cap.release()
        raise ValueError('Video has unreadable or invalid timing metadata')
    try:
        model = YOLO(str(weights))
    except Exception:
        cap.release()
        raise
    if model.task != 'segment':
        cap.release()
        raise ValueError('Expected the AlphaDent segmentation checkpoint')
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
                prediction = model.predict(frame, conf=confidence, imgsz=640, device=device,
                                           verbose=False, save=False)[0]
                frame_observations = []
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
                if frame_observations:
                    # Only annotate model-predicted source frames; the original video stays available.
                    annotated = prediction.plot()
                    if not cv2.imwrite(str(evidence / f'frame-{index:08d}.jpg'), annotated):
                        raise ValueError('Could not save source-frame evidence')
                    observations.extend(frame_observations)
            if decoded != int(expected_frames):
                raise ValueError(f'Incomplete video decode: {decoded}/{int(expected_frames)} frames')
            duration = decoded / fps
            highlights = pick_highlights(observations, duration)
            with weights.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            report = {
                'schema_version': 'dentilens-dental-1', 'video': video.name,
                'duration_sec': duration, 'fps': fps, 'frames_decoded': decoded,
                'sampled_frames': sampled, 'effective_sample_fps': fps / step,
                'model': {'name': 'AlphaDent dental-photo segmentation', 'weights_sha256': digest,
                          'classes': model.names, 'confidence_threshold': confidence, 'device': device},
                'observations': observations, 'highlights': highlights,
                'highlight_duration_sec': sum(clip['end_sec']-clip['start_sec'] for clip in highlights),
                'capabilities': {'visual_observations': True, 'procedure_action_recognition': False,
                                 'individual_tooth_numbering': False},
                'limitations': ['Trained on still dental photographs; video performance needs evaluation.',
                                'CFR video timing required. Fast events may fall between sampled frames.',
                                'Observation counts include repeated detections across frames.',
                                'Highlight video uses original frames and omits audio.'],
                'review_notice': 'Research prototype. All model observations require professional review.'}
            if highlights:
                export_clips(video, stage / 'highlights.mp4', highlights)
            write_json(stage / 'report.json', report)
            stage.rename(output)
            return report
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description='Analyze a dental video with a trained intraoral-photo model')
    parser.add_argument('video', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--weights', type=Path, default=Path(__file__).resolve().parents[1] / 'models/alphadent.pt')
    parser.add_argument('--sample-fps', type=float, default=1.0)
    parser.add_argument('--confidence', type=float, default=0.35)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    try:
        report = analyze(args.video, args.weights, args.output, args.sample_fps, args.confidence, args.device)
    except Exception as exc:
        parser.exit(1, f'Analysis failed: {exc}\n')
    print(f"Analyzed {report['sampled_frames']} frames; {len(report['observations'])} visual observations")
    print(f"Report saved to {args.output / 'report.json'}")


if __name__ == '__main__':
    main()
