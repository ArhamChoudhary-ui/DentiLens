"""Explainable greedy coverage selection; not claimed optimal or novel."""
import math

from .temporal import tiou


def select(candidates, duration, ratio=0.15, strategy='diversity', ensure_highlight=False, evidence=None):
    if not math.isfinite(duration) or duration <= 0 or not 0 < ratio <= 0.15:
        raise ValueError("Duration must be positive; budget ratio must be in (0, .15]")
    if strategy not in {'diversity', 'confidence', 'uniform', 'motion'}:
        raise ValueError("Unknown selector")
    budget = duration * ratio
    remaining = []
    for candidate in candidates:
        c = dict(candidate)
        if not 0 <= c['start'] < c['end'] <= duration:
            raise ValueError("Candidate outside video")
        if not all(math.isfinite(c.get(k, 0)) and 0 <= c.get(k, 0) <= 1
                   for k in ('confidence', 'visual', 'interaction')):
            raise ValueError("Invalid candidate score")
        if c['end'] - c['start'] < 4.0 - 1e-9:
            continue
        remaining.append(c)
    if ensure_highlight and duration < 4:
        raise ValueError('A four-second highlight requires at least four seconds of footage')
    if ensure_highlight and not remaining:
        remaining.append({'start': 0.0, 'end': 4.0, 'fallback': True})
    if ensure_highlight and budget < 4.0:
        # The guide requires >=4s and treats budget excess as a warning/penalty.
        # The three manifest videos shorter than 26.67s cannot satisfy both.
        budget = 4.0
        remaining = [{**c, 'end': c['start'] + 4.0} for c in remaining]
    selected, covered = [], set()
    while remaining:
        def utility(c):
            if strategy == 'confidence':
                return c.get('confidence', 0)
            if strategy == 'motion':
                return c.get('visual', 0)
            if strategy == 'uniform':
                # Farthest-first distribution across the original timeline.
                mid = (c['start'] + c['end']) / 2
                return min([abs(mid-(s['start']+s['end'])/2)/duration for s in selected]
                           or [1 - abs(mid-duration/2)/duration])
            pair = (c.get('verb'), c.get('noun'))
            novelty = float(all(tiou(c, s) == 0 for s in selected))
            new_action = float(pair != (None, None) and pair not in covered)
            redundancy = float(pair in covered)
            return max(0., min(1., .45*c.get('confidence', 0) + .30*new_action
                               + .10*novelty + .10*c.get('visual', 0)
                               + .05*c.get('interaction', 0) - .15*redundancy))
        ranked = sorted(remaining, key=lambda c: (-utility(c)/(c['end']-c['start']), c['start']))
        chosen = None
        for c in ranked:
            if c['end']-c['start'] <= budget + 1e-9 and all(
                c['end'] <= s['start'] or c['start'] >= s['end'] for s in selected
            ):
                chosen = c
                break
        if chosen is None:
            break
        importance = utility(chosen)
        remaining.remove(chosen)
        selected.append({**chosen, 'importance': importance})
        covered.add((chosen.get('verb'), chosen.get('noun')))
        budget -= chosen['end']-chosen['start']
    selected.sort(key=lambda c: c['start'])
    if evidence is not None:
        evidence.extend({key: c[key] for key in ('verb', 'noun', 'confidence', 'visual', 'importance', 'fallback')
                         if key in c and c[key] is not None} for c in selected)
    return [{'start': c['start'], 'end': c['end'], 'importance': c['importance']} for c in selected]
