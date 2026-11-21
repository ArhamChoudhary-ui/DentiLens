"""Offline Whisper inference on the browser-normalized source timeline."""
from functools import lru_cache
import subprocess
from time import perf_counter

import imageio_ffmpeg
import numpy as np

LANGUAGES = {'en': 'english', 'hi': 'hindi', 'es': 'spanish', 'fr': 'french', 'de': 'german'}


def audio_ready(models):
    root = models / 'whisper-tiny'
    return all((root / name).is_file() for name in ('model.safetensors', 'tokenizer.json', 'preprocessor_config.json'))


def extract_audio(source):
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-loglevel', 'error', '-i', str(source),
         '-map', '0:a:0', '-vn', '-af', 'aresample=async=1:first_pts=0', '-ar', '16000',
         '-ac', '1', '-f', 'f32le', 'pipe:1'],
        capture_output=True, timeout=180)
    if result.returncode:
        stderr = result.stderr.decode(errors='replace')
        if 'matches no streams' in stderr:
            return None
        raise ValueError('Audio could not be extracted from this recording.')
    return np.frombuffer(result.stdout, dtype='<f4').copy()


@lru_cache(maxsize=1)
def load_whisper(root):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    torch.set_num_threads(min(4, torch.get_num_threads()))
    processor = AutoProcessor.from_pretrained(root, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(root, local_files_only=True, use_safetensors=True)
    return pipeline('automatic-speech-recognition', model=model, tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor, device=-1, chunk_length_s=25)


def transcribe(source, models, language, duration, decoder=None):
    start = perf_counter()
    audio = extract_audio(source)
    if audio is None:
        return dict(state='no_audio', segments=[], language=language, model='whisper-tiny', seconds=0)
    if not len(audio) or float(np.max(np.abs(audio))) < .001:
        return dict(state='silent', segments=[], language=language, model='whisper-tiny', seconds=0)
    if decoder is None:
        if not audio_ready(models):
            raise ValueError('Install the local Whisper model before transcribing audio.')
        decoder = load_whisper(models / 'whisper-tiny')
    options = {'task':'transcribe'}
    if language in LANGUAGES:
        options['language'] = LANGUAGES[language]
    result = decoder({'array':audio, 'sampling_rate':16000}, return_timestamps=True, generate_kwargs=options)
    segments = []
    for chunk in result.get('chunks', []):
        text = chunk.get('text', '').strip()
        onset, end = chunk['timestamp']
        if not text or onset is None:
            continue
        onset=max(0,min(float(onset),duration))
        end=max(onset,min(duration,float(end) if end is not None else len(audio)/16000))
        if end>onset:
            segments.append(dict(id='audio-'+str(len(segments)), start=round(onset,3), end=round(end,3), text=text))
    return dict(state='ready', model='whisper-tiny', language=language, segments=segments,
                seconds=round(perf_counter()-start,2))
