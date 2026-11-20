"""Export local video frames and a manifest for tooth and instrument annotation."""
import argparse
import json
from pathlib import Path

import cv2


def export(videos, output, interval):
    output.mkdir(parents=True)
    manifest = []
    for clip, video in enumerate(videos):
        capture = cv2.VideoCapture(str(video))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not capture.isOpened() or fps <= 0:
                raise ValueError(f'Cannot read {video}')
            frame_index = 0
            step = max(1, round(fps * interval))
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame_index % step == 0:
                    height, width = frame.shape[:2]
                    scale = min(1, 960 / max(height, width))
                    resized = cv2.resize(frame, (round(width * scale), round(height * scale)))
                    filename = f'clip-{clip:02d}-frame-{frame_index:08d}.jpg'
                    if not cv2.imwrite(str(output / filename), resized):
                        raise OSError(f'Cannot write {filename}')
                    manifest.append(dict(image=filename, source=video.name,
                                         source_frame=frame_index, timestamp_sec=frame_index/fps,
                                         review_status='unlabeled'))
                frame_index += 1
        finally:
            capture.release()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {len(manifest)} frames across {len(videos)} clips')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('videos', type=Path, nargs='+')
    parser.add_argument('--interval', type=float, default=3)
    args = parser.parse_args()
    if not 0 < args.interval < float('inf'):
        parser.error('Interval must be positive and finite')
    export(args.videos, args.output, args.interval)
