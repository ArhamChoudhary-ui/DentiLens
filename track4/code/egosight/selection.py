"""Explainable greedy coverage selection; not claimed optimal or novel."""
import math
from typing import Any, Dict, List, Optional

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

            mid = (c['start'] + c['end']) / 2.0
            temporal_diversity = (
                min([abs(mid - (s['start'] + s['end']) / 2.0) / duration for s in selected])
                if selected else (1.0 - abs(mid - duration / 2.0) / duration)
            )
            # Intro penalty: static or low-motion title cards in first 3.5 seconds
            intro_penalty = 0.25 if (c['start'] < 3.5 and c.get('visual', 0.0) < 0.035 and pair in ((None, None), ('other', 'other'))) else 0.0

            base = (0.40 * c.get('confidence', 0)
                    + 0.25 * new_action
                    + 0.10 * novelty
                    + 0.10 * c.get('visual', 0)
                    + 0.10 * min(1.0, temporal_diversity * 2.0)
                    + 0.05 * c.get('interaction', 0)
                    - 0.15 * redundancy
                    - intro_penalty)
            return max(0., min(1., base))

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


def explain_highlights(
    highlights: List[Dict[str, Any]],
    actions: Optional[List[Dict[str, Any]]] = None,
    activity: str = 'activity',
    scenes: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate transparent, multi-signal explanations and proposal linkages for highlights.
    """
    actions = actions or []
    details = []
    activity_clean = activity.replace('_', ' ')

    for h in highlights:
        s, e = h['start'], h['end']
        duration = round(e - s, 2)
        # Find overlapping proposal
        matching = None
        best_overlap = 0.0
        for idx, act in enumerate(actions, start=1):
            overlap = max(0.0, min(e, act['end']) - max(s, act['start']))
            if overlap > best_overlap:
                best_overlap = overlap
                matching = (idx, act)

        related_idx = matching[0] if matching else None
        related_act = matching[1] if matching else None

        # Determine rationale
        is_rec = related_act and (
            related_act.get('verb', 'other') != 'other' or related_act.get('noun', 'other') != 'other'
        )
        conf_pct = f"{round(h.get('importance', 0.5) * 100)}%"

        if is_rec:
            verb, noun = related_act['verb'], related_act['noun']
            rat = (
                f"Selected for high-confidence recognized interaction “{verb} {noun}” "
                f"({conf_pct} candidate score), prioritized under the 15% budget."
            )
        elif related_act and related_act.get('visual', 0) > 0.05:
            rat = (
                f"Selected for peak {activity_clean} movement energy "
                f"({round(related_act.get('visual', 0), 2)} visual motion) and temporal model activation ({conf_pct})."
            )
        else:
            rat = (
                f"Ranked as a salient {activity_clean} moment with high spatiotemporal feature energy "
                f"({conf_pct}), satisfying non-overlapping temporal budget constraints."
            )

        details.append({
            'start': s,
            'end': e,
            'duration': duration,
            'importance': round(h.get('importance', 0.0), 4),
            'related_proposal_index': related_idx,
            'related_action': f"{related_act['verb']} {related_act['noun']}" if is_rec else None,
            'rationale': rat,
        })
    return details
