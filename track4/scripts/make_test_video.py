"""Synthetic geometric fixture for decoding tests, never semantic validation."""
import argparse
from pathlib import Path
import cv2
import numpy as np


def make_video(path, seconds=120, fps=10):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(96,64))
    if not writer.isOpened():
        raise RuntimeError('MP4 encoder unavailable')
    try:
        for i in range(int(seconds*fps)):
            frame=np.zeros((64,96,3),dtype=np.uint8)
            # Vary motion over time to exercise ranking without implying an action.
            x=(i*(1+(i//100)%4))%76
            frame[20:40,x:x+20]=(255,160,60)
            writer.write(frame)
    finally:
        writer.release()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--seconds',type=float,default=120)
    a=p.parse_args()
    make_video(a.output,a.seconds)
