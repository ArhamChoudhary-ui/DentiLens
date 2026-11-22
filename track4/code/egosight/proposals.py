"""Intelligent temporal action proposal generation, quality ranking, and evidence-grounded categorization."""
from typing import Any, Dict, List, Tuple
import numpy as np


def tiou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Calculate temporal Intersection over Union between two intervals."""
    s_a, e_a = a['start'], a['end']
    s_b, e_b = b['start'], b['end']
    intersection = max(0.0, min(e_a, e_b) - max(s_a, s_b))
    union = max(e_a, e_b) - min(s_a, s_b)
    return intersection / union if union > 0.0 else 0.0


def categorize_proposal(
    verb: str,
    noun: str,
    act_conf: float,
    motion: float,
    d_score: float,
    d_motion: float,
) -> str:
    """Categorize proposal strictly grounded in available evidence."""
    if verb != 'other' or noun != 'other':
        return 'Recognized action'
    if (d_score >= 0.15 or d_motion >= 0.06) and act_conf >= 0.12:
        return 'Activity transition'
    if act_conf >= 0.30 and motion >= 0.03:
        return 'Potential activity interval'
    if motion < 0.02 or act_conf < 0.20:
        return 'Visually unclear interval'
    return 'Potential activity interval'


def proposal_rationale(
    category: str,
    act_conf: float,
    motion: float,
    d_score: float,
    activity: str,
    verb: str = 'other',
    noun: str = 'other',
) -> str:
    pct = f"{round(act_conf * 100)}%"
    if category == 'Recognized action':
        if verb != 'other' or noun != 'other':
            return f"Recognized action “{verb} {noun}” with {pct} confidence."
        return f"Trained action head recognized with {pct} confidence."
    if category == 'Activity transition':
        return f"Activity transition ({pct} {activity} confidence, Δscore {round(d_score, 2)}) marking activity shift."
    if category == 'Potential activity interval':
        return f"Salient {activity} movement with {pct} model confidence and active visual motion."
    if category == 'Visually unclear interval':
        return f"Marginal or low-motion interval ({pct} confidence, {round(motion, 3)} visual motion)."
    return f"Temporal proposal identified with {pct} model confidence."


def process_proposals(
    window_scores: List[Tuple[float, float, Dict[str, float], float]],
    duration: float,
    label: str,
    min_conf: float = 0.10,
    iou_thresh: float = 0.30,
    custom_actions: List[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process raw window predictions into deduplicated, ranked, and categorized action proposals.

    Returns:
        actions: list of official Track 4 action proposals (chronological)
        action_metadata: list of rich proposal metadata (category, quality_rank, rationale)
        proposal_stats: summary statistics of candidate generation, retention, and suppression
    """
    if custom_actions is not None:
        # EPIC-KITCHENS or trained model mode with existing genuine verb-noun predictions
        retained = []
        metadata = []
        for i, a in enumerate(sorted(custom_actions, key=lambda x: (x['start'], x['end']))):
            verb = a.get('verb', 'other')
            noun = a.get('noun', 'other')
            cat = categorize_proposal(verb, noun, a.get('confidence', 0.0), a.get('visual', 0.0), 0.0, 0.0)
            rat = proposal_rationale(cat, a.get('confidence', 0.0), a.get('visual', 0.0), 0.0, label, verb, noun)
            item = dict(a)
            retained.append(item)
            metadata.append({
                'start': item['start'],
                'end': item['end'],
                'category': cat,
                'quality_rank': i + 1,
                'quality_score': float(a.get('confidence', 0.0)),
                'rationale': rat,
            })
        stats = {
            'raw_candidates': len(custom_actions),
            'retained': len(retained),
            'suppressed_nms': 0,
            'suppressed_low_conf': 0,
            'suppressed_low_info': 0,
            'rejection_reasons': {
                'redundant_nms_overlap': 0,
                'low_confidence': 0,
                'low_information_static': 0,
            },
        }
        return retained, metadata, stats

    if not window_scores:
        return [], [], {
            'raw_candidates': 0,
            'retained': 0,
            'suppressed_nms': 0,
            'suppressed_low_conf': 0,
            'suppressed_low_info': 0,
            'rejection_reasons': {
                'redundant_nms_overlap': 0,
                'low_confidence': 0,
                'low_information_static': 0,
            },
        }

    # Step 1: Compute temporal gradients and statistics across consecutive windows
    candidates = []
    suppressed_low_conf = 0
    suppressed_low_info = 0
    n = len(window_scores)

    conf_values = [scores.get(label, 0.0) if label != 'other' else max(scores.values(), default=0.0)
                   for _, _, scores, _ in window_scores]
    mean_conf = float(np.mean(conf_values)) if conf_values else 0.0

    for i, (start, end, scores, motion) in enumerate(window_scores):
        act_conf = scores.get(label, 0.0) if label != 'other' else max(scores.values(), default=0.0)

        # Calculate temporal gradients
        prev_conf = window_scores[i - 1][2].get(label, act_conf) if i > 0 else act_conf
        next_conf = window_scores[i + 1][2].get(label, act_conf) if i < n - 1 else act_conf
        d_score = float(max(abs(act_conf - prev_conf), abs(next_conf - act_conf)))

        prev_motion = window_scores[i - 1][3] if i > 0 else motion
        next_motion = window_scores[i + 1][3] if i < n - 1 else motion
        d_motion = float(max(abs(motion - prev_motion), abs(motion - next_motion)))

        # Suppression criteria:
        # A) Raw threshold
        if act_conf < min_conf and motion < 0.05:
            suppressed_low_conf += 1
            continue

        # B) Low-information section: marginal score, low motion, flat gradient
        if act_conf < 0.18 and motion < 0.025 and d_score < 0.10:
            suppressed_low_info += 1
            continue

        # Quality scoring grounded in confidence, visual motion, and temporal contrast
        distinctiveness = abs(act_conf - mean_conf) / max(0.05, mean_conf)
        q_score = float(
            0.50 * act_conf +
            0.30 * min(1.0, motion * 10.0) +
            0.20 * min(1.0, d_score * 3.0)
        )
        cat = categorize_proposal('other', 'other', act_conf, motion, d_score, d_motion)
        rat = proposal_rationale(cat, act_conf, motion, d_score, label)

        candidates.append({
            'start': start,
            'end': end,
            'verb': 'other',
            'noun': 'other',
            'confidence': float(np.clip(act_conf, 0.0, 1.0)),
            'visual': float(motion),
            'quality_score': q_score,
            'distinctiveness': distinctiveness,
            'd_score': d_score,
            'd_motion': d_motion,
            'category': cat,
            'rationale': rat,
        })

    # Step 2: Temporal Non-Maximum Suppression (eliminate redundant overlapping sliding windows)
    candidates.sort(key=lambda c: (-c['quality_score'], c['start']))
    retained_candidates = []
    suppressed_nms = 0

    for cand in candidates:
        if any(tiou(cand, kept) > iou_thresh for kept in retained_candidates):
            suppressed_nms += 1
        else:
            retained_candidates.append(cand)

    # Step 3: Sort chronologically and assign quality ranks
    retained_by_quality = sorted(retained_candidates, key=lambda c: -c['quality_score'])
    for rank, cand in enumerate(retained_by_quality, start=1):
        cand['quality_rank'] = rank

    retained_chronological = sorted(retained_candidates, key=lambda c: (c['start'], c['end']))

    actions = []
    action_metadata = []
    for cand in retained_chronological:
        actions.append({
            'start': cand['start'],
            'end': cand['end'],
            'verb': cand['verb'],
            'noun': cand['noun'],
            'confidence': cand['confidence'],
            'visual': cand['visual'],
        })
        action_metadata.append({
            'start': cand['start'],
            'end': cand['end'],
            'category': cand['category'],
            'quality_rank': cand['quality_rank'],
            'quality_score': round(cand['quality_score'], 4),
            'rationale': cand['rationale'],
            'd_score': round(cand['d_score'], 4),
            'd_motion': round(cand['d_motion'], 4),
        })

    stats = {
        'raw_candidates': len(window_scores),
        'retained': len(actions),
        'suppressed_nms': suppressed_nms,
        'suppressed_low_conf': suppressed_low_conf,
        'suppressed_low_info': suppressed_low_info,
        'rejection_reasons': {
            'redundant_nms_overlap': suppressed_nms,
            'low_confidence': suppressed_low_conf,
            'low_information_static': suppressed_low_info,
        },
    }
    return actions, action_metadata, stats
