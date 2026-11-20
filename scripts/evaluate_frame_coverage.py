"""Measure coverage against independent annotations of the rendered timeline."""
import argparse
from collections import Counter
import json
from pathlib import Path

VISIBLE = {'clear', 'partially_visible'}
VISIBILITY = VISIBLE | {'absent', 'occluded', 'blurred', 'poor_lighting', 'camera_motion', 'uncertain'}


def measure(timeline_path, labels_path):
    with Path(timeline_path).open() as stream:
        timeline = [json.loads(line) for line in stream]
    labels = json.loads(Path(labels_path).read_text())
    if not isinstance(labels, list) or len(labels) != len(timeline):
        raise ValueError('Give one visibility label per frame in the rendered video.')
    causes = Counter()
    eligible = covered = individual = 0
    for index, (entry, label) in enumerate(zip(timeline, labels)):
        if entry['frame'] != index or not isinstance(label, dict) or label.get('frame') != index:
            raise ValueError(f'Expected frame {index} in both inputs.')
        visibility, is_eligible = label.get('visibility'), label.get('eligible')
        if visibility not in VISIBILITY or type(is_eligible) is not bool:
            raise ValueError(f'Frame {index} needs a visibility reason and an explicit eligibility decision.')
        if is_eligible and visibility in {'absent', 'uncertain'}:
            raise ValueError(f'Frame {index} cannot be eligible with visibility {visibility}.')
        dental = [row for row in entry['annotations'] if row['category'] in ('tooth_identity', 'tooth_region')]
        eligible += is_eligible
        covered += is_eligible and bool(dental)
        individual += is_eligible and any(row['category'] == 'tooth_identity' for row in dental)
        if not dental:
            causes[f'{visibility}: {entry.get("reason", "blank")}'] += 1
    return dict(frames=len(timeline), eligible_frames=eligible, annotated_eligible_frames=covered,
                individual_tooth_eligible_frames=individual, coverage=covered / eligible if eligible else None,
                individual_tooth_coverage=individual / eligible if eligible else None, blank_frames=dict(causes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('timeline')
    parser.add_argument('visibility_labels')
    args = parser.parse_args()
    try:
        print(json.dumps(measure(args.timeline, args.visibility_labels), indent=2))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
