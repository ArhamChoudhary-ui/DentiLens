"""Real VideoMAE embeddings plus separately trained verb/noun/category heads."""
from pathlib import Path
import numpy as np
import torch

ENCODER = 'MCG-NJU/videomae-base'
ENCODER_REVISION = 'dc740ceda42fce44faed2ea03c6d447db72f6af9'


def device_for(requested='auto'):
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if requested not in {'cpu','cuda','mps'}:
        raise ValueError('Unknown compute device')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable')
    return requested


class Encoder:
    def __init__(self, source=ENCODER, revision=ENCODER_REVISION, device='auto', offline=True):
        from transformers import VideoMAEImageProcessor, VideoMAEModel
        self.device = device_for(device)
        args = dict(local_files_only=offline, revision=revision)
        self.processor = VideoMAEImageProcessor.from_pretrained(source, **args)
        self.model, info = VideoMAEModel.from_pretrained(
            source, use_safetensors=True, output_loading_info=True, **args)
        if info['missing_keys'] or info.get('mismatched_keys'):
            raise ValueError('Encoder has uninitialized or incompatible weights')
        self.model.to(self.device).eval()
        self.source = source
        self.revision = getattr(self.model.config, '_commit_hash', None) or revision
        self.frames = self.model.config.num_frames
        self.dimension = self.model.config.hidden_size

    def encode(self, frames):
        values = self.processor(frames, return_tensors='pt')['pixel_values'].to(self.device)
        with torch.inference_mode():
            output = self.model(pixel_values=values).last_hidden_state.mean(dim=1)
        output = output.cpu().float()
        if not torch.isfinite(output).all():
            raise ValueError('Nonfinite encoder output')
        return output[0]


class Predictor:
    def __init__(self, checkpoint, device='auto', offline=True, encoder_path=None):
        if not Path(checkpoint).is_file():
            raise ValueError('Trained heads missing. Run scripts/train.py with licensed labeled footage first.')
        self.artifact = torch.load(checkpoint, map_location='cpu', weights_only=True)
        a = self.artifact
        if a.get('format') != 'dentilens-linear-v1':
            raise ValueError('Unsupported checkpoint format')
        from .contract import StudentKit
        kit = StudentKit()
        kit.validate_classes(a['classes'])
        for name in ('verb','noun','label'):
            if not a['classes'][name] or len(set(a['classes'][name])) != len(a['classes'][name]):
                raise ValueError('Invalid trained class vocabulary')
        if not a.get('pairs') or any(
            len(pair) != 2 or pair[0] not in a['classes']['verb'] or pair[1] not in a['classes']['noun']
            for pair in a['pairs']
        ):
            raise ValueError('Invalid trained verb–noun pairs')
        self.encoder = Encoder(encoder_path or a['encoder'], a['revision'], device, offline)
        self.heads = {}
        for name, classes in a['classes'].items():
            head = torch.nn.Linear(self.encoder.dimension, len(classes))
            head.load_state_dict(a['heads'][name])
            head.eval()
            self.heads[name] = head

    def scores(self, reader, intervals):
        rows = {k: [] for k in self.heads}
        visual = []
        for start, end in intervals:
            frames, _ = reader.sample(start, end, self.encoder.frames)
            feature = self.encoder.encode(frames)
            motion = np.mean([np.abs(b.astype(np.float32)-a.astype(np.float32)).mean()/255
                              for a,b in zip(frames, frames[1:])])
            visual.append(float(motion))
            with torch.inference_mode():
                for name, head in self.heads.items():
                    rows[name].append(head(feature).softmax(dim=-1).numpy())
        return {**{k: np.stack(v) for k,v in rows.items()}, 'visual': visual}
