"""CFR MP4 decoding. Samples are RGB and timestamped in source seconds."""
from dataclasses import asdict, dataclass
from pathlib import Path
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    fps: float
    frames: int
    width: int
    height: int

    def to_dict(self):
        return asdict(self)


class VideoReader:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_file() or self.path.suffix.lower() != ".mp4":
            raise ValueError("Input must be an existing MP4 file")
        cap = cv2.VideoCapture(str(self.path))
        try:
            if not cap.isOpened():
                raise ValueError("Video cannot be opened")
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if any(not math.isfinite(x) or x <= 0 for x in (fps, count, width, height)):
                raise ValueError("Invalid video metadata")
            self.info = VideoInfo(count / fps, fps, int(count), int(width), int(height))
        finally:
            cap.release()

    def verify(self):
        """Full sequential decode catches corrupt/truncated media before export."""
        cap = cv2.VideoCapture(str(self.path))
        decoded = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame is None or frame.shape[:2] != (self.info.height, self.info.width):
                    raise ValueError("Invalid decoded frame or changing resolution")
                decoded += 1
        finally:
            cap.release()
        if decoded != self.info.frames:
            raise ValueError(f"Incomplete decode: expected {self.info.frames}, got {decoded}")

    def sample(self, start, end, count=16, size=224):
        if not (0 <= start < end <= self.info.duration) or count < 1 or size < 1:
            raise ValueError("Invalid sample interval or dimensions")
        first = min(int(start * self.info.fps), self.info.frames - 1)
        last = min(max(first, math.ceil(end * self.info.fps) - 1), self.info.frames - 1)
        indices = np.linspace(first, last, count).round().astype(int)
        cap = cv2.VideoCapture(str(self.path))
        frames = []
        try:
            for idx in indices:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)):
                    raise ValueError(f"Cannot seek to source frame {idx}")
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode source frame {idx}")
                # Resize with preserved aspect ratio; encoder applies its own crop.
                h, w = frame.shape[:2]
                scale = size / min(h, w)
                frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return frames, (indices / self.info.fps).tolist()


def windows(duration, length=4.0, stride=2.0):
    if any(not math.isfinite(x) or x <= 0 for x in (duration, length, stride)) or stride > length:
        raise ValueError("Require finite positive duration and 0 < stride <= length")
    start = 0.0
    while start < duration:
        end = min(duration, start + length)
        yield start, end
        if end >= duration:
            break
        start += stride
