"""Export original-source frames only. FFmpeg re-encodes accurate boundaries."""
from pathlib import Path
import shutil
import subprocess
import tempfile


def export_highlights(source, destination, highlights):
    executable = shutil.which('ffmpeg')
    if not executable:
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise ValueError('MP4 export requires FFmpeg or the imageio-ffmpeg package')
    if not highlights:
        raise ValueError('There are no highlights to export')
    destination = Path(destination).resolve()
    source = Path(source).resolve()
    if destination == source:
        raise ValueError('Export must not overwrite the original video')
    destination.parent.mkdir(parents=True, exist_ok=True)
    filters, labels = [], []
    for i,h in enumerate(highlights):
        filters.append(f"[0:v]trim=start={h['start']}:end={h['end']},setpts=PTS-STARTPTS[v{i}]")
        labels.append(f'[v{i}]')
    filters.append(''.join(labels)+f'concat=n={len(labels)}:v=1:a=0[out]')
    # Video-only export: audio preservation is not implemented in this baseline.
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        staged = Path(tmp)/'highlight.mp4'
        command = [executable,'-v','error','-nostdin','-y','-i',str(source),'-filter_complex',
                   ';'.join(filters),'-map','[out]','-c:v','libx264','-pix_fmt','yuv420p',
                   '-movflags','+faststart',str(staged)]
        subprocess.run(command, check=True, capture_output=True, timeout=600)
        staged.replace(destination)
