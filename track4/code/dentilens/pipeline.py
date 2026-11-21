from pathlib import Path
import numpy as np

from .video import VideoReader, windows
from .temporal import smooth, nms
from .selection import select
from .schema import validate


def analyze(path, mode='model', checkpoint=None, device='auto', offline=True,
            allow_short=False, strategy='diversity', encoder_path=None, dataset=None, weights_root=None):
    reader = VideoReader(path)
    duration = reader.info.duration
    if duration < 4:
        raise ValueError('At least four seconds of source footage are required')
    if mode not in {'model', 'motion-baseline', 'infrastructure'}:
        raise ValueError('Unknown inference mode')
    # Fail before expensive decoding when inference prerequisites are absent.
    published = mode == 'model' and checkpoint is None and dataset is not None
    if mode == 'model' and not published and (not checkpoint or not Path(checkpoint).is_file()):
        raise ValueError('No trained action checkpoint. Train with scripts/train.py; use an explicit baseline for infrastructure QA.')
    if not (published and dataset=='epic_kitchens_100'):
        reader.verify()
    intervals = list(windows(duration, 4, 2))
    actions, candidates = [], []
    label = 'unclassified'  # Explicit abstention in development modes, never a predicted class.
    provenance = {'mode': mode, 'video': reader.info.to_dict(), 'official_schema_verified': False,
                  'dental': {'enabled': False, 'reason': 'No trained dental weights supplied'},
                  'limitations': ['CFR MP4 required; transcode VFR footage before analysis.',
                                  'Window-level localization baseline; no trained boundary head.',
                                  'No validated hand/object detector; interaction scoring is disabled.']}
    if published:
        from .pretrained import EpicTSN, FPVTransfer
        root = Path(weights_root or Path(__file__).resolve().parents[2] / 'weights')
        predictor = EpicTSN(root,device) if dataset=='epic_kitchens_100' else FPVTransfer(root,device)
        actions,candidates,label,score,details = predictor.predict(reader)
        actions = nms(actions)
        actions = sorted(sorted(actions,key=lambda action:-action['confidence'])[:2000],
                         key=lambda action:(action['start'],action['end']))
        provenance.update(**details,label_score=score,action_scores=actions,device=predictor.device,dataset=dataset)
        if dataset=='epic_kitchens_100':
            provenance['class_contract'] = 'EPIC category prescribed by the organizer guide'
    elif mode == 'model':
        from .model import Predictor
        predictor = Predictor(checkpoint, device, offline, encoder_path)
        scores = predictor.scores(reader, intervals)
        verbs, nouns = smooth(scores['verb']), smooth(scores['noun'])
        classes = predictor.artifact['classes']
        pairs = sorted({tuple(pair) for pair in predictor.artifact['pairs']})
        verb_ids = {name:i for i,name in enumerate(classes['verb'])}
        noun_ids = {name:i for i,name in enumerate(classes['noun'])}
        pair_verb_ids = [verb_ids[v] for v,n in pairs]
        pair_noun_ids = [noun_ids[n] for v,n in pairs]
        pair_scores = verbs[:,pair_verb_ids] * nouns[:,pair_noun_ids]
        # Choose only verb–noun pairs observed in training; no fabricated mappings.
        for i, (start, end) in enumerate(intervals):
            best = int(pair_scores[i].argmax())
            verb,noun = pairs[best]
            confidence = float(pair_scores[i,best])
            if confidence >= .05:
                actions.append(dict(verb=verb, noun=noun, start=start, end=end, confidence=confidence,
                                    visual=scores['visual'][i]))
        actions = nms(actions)
        actions = sorted(sorted(actions, key=lambda action: -action['confidence'])[:2000],
                         key=lambda action: (action['start'], action['end']))
        candidates = [dict(a) for a in actions]
        label_probabilities = np.mean(scores['label'], axis=0)
        label_index = int(label_probabilities.argmax())
        label = classes['label'][label_index]
        provenance['label_score'] = float(label_probabilities[label_index])
        provenance.update(encoder=predictor.artifact['encoder'], revision=predictor.artifact['revision'],
                          classes_source='training annotations checked against supplied supported_classes.json',
                          action_scores=actions, validation=predictor.artifact['validation'])
    elif mode == 'motion-baseline':
        for start, end in intervals:
            frames, _ = reader.sample(start, end, count=4, size=64)
            motion = float(np.mean([np.abs(b.astype(float)-a.astype(float)).mean()/255
                                    for a,b in zip(frames, frames[1:])]))
            candidates.append(dict(start=start, end=end, visual=motion, confidence=0.0))
        strategy = 'motion'
        provenance['limitations'].append('Motion-only baseline: no recognized actions or predicted video category.')
    highlights = select(candidates, duration, strategy=strategy, ensure_highlight=True)
    if not candidates:
        provenance['highlight_fallback'] = 'Uniform first four seconds because no scored candidates were available'
    if duration * 0.15 < 4.0:
        provenance['budget_warning'] = 'Four-second minimum exceeds 15% for this short source video'
    result = dict(video_id=Path(path).stem, label=label,
                  actions=[{k:a[k] for k in ('verb','noun','start','end')} for a in actions],
                  highlights=highlights)
    validate(result, duration)
    provenance['highlight_seconds'] = sum(h['end']-h['start'] for h in highlights)
    provenance['duration'] = duration
    provenance['candidates'] = candidates
    return result, provenance
