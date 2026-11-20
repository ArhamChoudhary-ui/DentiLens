"""Run the tooth name, number and cavity models against the same source frame."""
from pathlib import Path

import cv2
from PIL import Image

from dental_modes import CAVITY_CLASSES, COMBINED_MODE, FDI_CLASSES, TOOTH_TYPES, fdi_name
from grounded_sam import GroundedBoxDetector
from tooth_sam import DentalBoxDetector, fdi_crop, overlap_fraction

IDENTITY_THRESHOLD = 0.35
CAVITY_THRESHOLD = 0.50


def intersection_over_union(first, second):
    overlap_first = overlap_fraction(first, second)
    overlap_second = overlap_fraction(second, first)
    if not overlap_first or not overlap_second:
        return 0.0
    return 1 / (1 / overlap_first + 1 / overlap_second - 1)


def combine_identities(number_rows, type_rows):
    """Associate two predictions for one tooth, preserving their separate scores."""
    strongest = {}
    for row in sorted(number_rows, key=lambda item: item['confidence'], reverse=True):
        strongest.setdefault(row['model_label'], row)
    numbers = list(strongest.values())
    pairs = []
    for number_index, number in enumerate(numbers):
        for type_index, tooth in enumerate(type_rows):
            overlap = intersection_over_union(number['bbox_xyxy_normalized'], tooth['bbox_xyxy_normalized'])
            if overlap >= 0.50:
                pairs.append((overlap, number_index, type_index))

    matches, used_types = {}, set()
    for _, number_index, type_index in sorted(pairs, key=lambda pair: -pair[0]):
        if number_index not in matches and type_index not in used_types:
            matches[number_index] = type_rows[type_index]
            used_types.add(type_index)

    findings = []
    for index, row in enumerate(numbers):
        code = row['model_label']
        tooth_type = TOOTH_TYPES[int(code[1]) - 1]
        candidate = matches.get(index)
        finding = {**row, 'category': 'tooth_identity', 'number_score': row['confidence']}
        if candidate:
            finding.update(type_candidate=candidate['model_label'], type_score=candidate['confidence'])
        if candidate and tooth_type != candidate['model_label']:
            finding.update(model_label='uncertain tooth identity', fdi_candidate=code, identity_status='conflicting')
        else:
            finding.update(fdi_number=code, tooth_name=fdi_name(code), tooth_type=tooth_type,
                           identity_status='type agreement' if candidate else 'number model only')
        findings.append(finding)

    for index, row in enumerate(type_rows):
        if index not in used_types:
            findings.append({**row, 'category': 'tooth_identity', 'tooth_type': row['model_label'],
                             'type_score': row['confidence'], 'identity_status': 'type model only'})
    return findings


def nearby_identity(cavity, teeth):
    """Return a tooth suggestion only when the flag overlaps one unambiguous box."""
    overlaps = [tooth for tooth in teeth if
                overlap_fraction(cavity['bbox_xyxy_normalized'], tooth['bbox_xyxy_normalized']) >= .25]
    if len(overlaps) != 1 or overlaps[0]['identity_status'] == 'conflicting':
        return None
    tooth = overlaps[0]
    if overlap_fraction(cavity['bbox_xyxy_normalized'], tooth['bbox_xyxy_normalized']) < .65:
        return None
    if 'fdi_number' in tooth:
        return f"FDI {tooth['fdi_number']} · {tooth['tooth_name']}"
    return tooth['tooth_type']


class CombinedReview:
    mode = COMBINED_MODE
    prompt = None
    model_name = 'DentiLens combined dental review'
    localization_method = 'Shared tooth region with separate tooth identity and cavity detectors'
    limitations = [
        'Tooth names, FDI numbers, and cavity flags are experimental suggestions. Video accuracy is unmeasured.',
        'A conflict between overlapping number and type predictions suppresses the exact FDI label.',
        'When only one identity model sees a tooth, its suggestion is shown for review.',
        'A cavity flag must overlap a predicted tooth region. Missing flags do not rule out decay.',
        'A nearby tooth label is a spatial association. Verify the tooth and the flag independently.',
        'Partial and mirrored footage can lead to wrong numbers or missing predictions.',
    ]

    def __init__(self, models, grounded_models, confidence=IDENTITY_THRESHOLD, device='cpu'):
        root = Path(models)
        self.locator = GroundedBoxDetector(grounded_models, device)
        self.number_model = DentalBoxDetector(root / 'tooth-fdi.pt', confidence, device, 'tooth-fdi')
        self.type_model = DentalBoxDetector(root / 'tooth-types.pt', confidence, device, 'tooth-types')
        self.cavity_model = DentalBoxDetector(root / 'cavity-review.pt', CAVITY_THRESHOLD, device, 'cavity-review')
        labels = (*FDI_CLASSES, *TOOTH_TYPES, CAVITY_CLASSES[0], 'uncertain tooth identity')
        self.names = dict(enumerate(labels))
        self.class_ids = {label: key for key, label in self.names.items()}
        self.provenance = {
            **self.locator.provenance,
            'tooth_fdi': self.number_model.provenance['detector'],
            'tooth_type': self.type_model.provenance['detector'],
            'cavity': self.cavity_model.provenance['detector'],
            'settings': {'identity_threshold': confidence, 'cavity_threshold': CAVITY_THRESHOLD,
                         'tooth_region_threshold': .50, 'identity_match_iou': .50,
                         'minimum_inside_tooth_region': .50},
        }
        self.observation_counts = dict(candidates_before_tooth_filter=0, rejected_outside_tooth_regions=0,
                                       conflicting_tooth_identities=0)

    def predict(self, source_frame, *, include_region=False, include_cavities=True):
        height, width = source_frame.shape[:2]
        scale = min(1.0, 960 / max(width, height))
        frame = cv2.resize(source_frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        regions = self.locator.detect(image, 'teeth.', .50)
        crop = fdi_crop(image, regions)
        if crop is None:
            return [], frame
        tooth_regions = [box.detach().cpu().tolist() for box, label in
                         zip(regions['boxes'], regions['text_labels']) if 'teeth' in label.lower()]

        numbers, _ = self.number_model.detect_boxes(frame, crop)
        types, type_boxes = self.type_model.detect_boxes(frame, crop)
        types = [row for row, box in zip(types, type_boxes)
                 if any(overlap_fraction(box, region) >= .50 for region in tooth_regions)]
        teeth = combine_identities(numbers, types)
        self.observation_counts['conflicting_tooth_identities'] += sum(
            row['identity_status'] == 'conflicting' for row in teeth)

        candidates, boxes = self.cavity_model.detect_boxes(frame) if include_cavities else ([], [])
        self.observation_counts['candidates_before_tooth_filter'] += len(candidates)
        findings = teeth[:]
        for row, box in zip(candidates, boxes):
            if not any(overlap_fraction(box, region) >= .50 for region in tooth_regions):
                self.observation_counts['rejected_outside_tooth_regions'] += 1
                continue
            row['category'] = 'cavity_candidate'
            tooth = nearby_identity(row, teeth)
            if tooth is not None:
                row['nearby_tooth_suggestion'] = tooth
            findings.append(row)

        for row in findings:
            row['model_class_id'] = self.class_ids[row['model_label']]
            color = ((70, 185, 255) if row['category'] == 'cavity_candidate' else
                     (220, 208, 104) if row.get('identity_status') == 'conflicting' else (70, 220, 110))
            x1, y1, x2, y2 = [round(value * extent) for value, extent in
                              zip(row['bbox_xyxy_normalized'], [image.width, image.height] * 2)]
            cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        if include_region and not teeth:
            choices = ((box, score) for box, score, label in
                       zip(regions['boxes'], regions['scores'], regions['text_labels']) if 'teeth' in label.lower())
            box, score = max(choices, key=lambda item: float(item[1]))
            xyxy = box.detach().cpu().tolist()
            bounds = [min(1, max(0, value / extent)) for value, extent in
                      zip(xyxy, [image.width, image.height] * 2)]
            if bounds[0] < bounds[2] and bounds[1] < bounds[3]:
                findings.insert(0, {'model_label': 'Visible tooth region', 'category': 'tooth_region',
                                    'confidence': float(score), 'bbox_xyxy_normalized': bounds})
        return findings, frame
