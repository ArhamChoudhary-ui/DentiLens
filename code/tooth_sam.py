"""Use the locally fine-tuned tooth detector to prompt SAM 2.1."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from dental_modes import TRAINED_MODES, fdi_name
from grounded_sam import GroundedBoxDetector, GroundedSegmenter


class ToothSegmenter:
    model_name = 'DentiLens intraoral YOLOv8n + SAM 2.1 tiny'
    prompt = None
    limitations = ['Experimental predictions require professional review.',
                   'Still-image detection metrics do not establish video performance.']

    def __init__(self, weights, model_directory, confidence=0.35, device='cpu', mode='tooth-sam2'):
        from ultralytics import YOLO

        weights, model_directory = Path(weights), Path(model_directory)
        self.confidence, self.device = confidence, device
        self.mode = mode
        self.model_name = 'DentiLens YOLOv8n · ' + TRAINED_MODES[mode][1]
        self.localization_method = 'Trained dental detector boxes'
        self.names = dict(enumerate(TRAINED_MODES[mode][2]))
        metadata = json.loads(weights.with_suffix('.json').read_text())
        with weights.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != metadata['weights_sha256']:
            raise ValueError('Detector weights do not match their provenance')
        self.detector = YOLO(str(weights))
        if self.detector.task != 'detect' or self.detector.names != self.names:
            raise ValueError('Detector classes differ from the selected mode')
        self.image_size = metadata['training']['image_size']
        self.provenance = {'detector': metadata}
        if mode == 'cavity-review':
            self.tooth_locator = GroundedBoxDetector(model_directory, device)
            self.provenance.update(self.tooth_locator.provenance)
            self.provenance['cavity_filter'] = {
                'tooth_prompt': 'teeth.', 'tooth_score_threshold': 0.5, 'minimum_inside_fraction': 0.5}
            self.observation_counts = {'candidates_before_tooth_filter': 0, 'rejected_outside_tooth_regions': 0}
            self.limitations = self.limitations + [
                'A flag is a suggestion for visual review. Clinical diagnosis requires professional assessment.',
                'A missing flag cannot rule out decay. Normal-appearance classes are suppressed in the interface.',
                'Flags must overlap a predicted tooth region. This can suppress true findings when teeth are missed.']
        elif mode == 'tooth-fdi':
            self.limitations = self.limitations + [
                'FDI numbers assume standard image orientation. Mirrored footage and partial views can lead to side or jaw errors.']
        elif mode == 'tooth-types':
            self.limitations = self.limitations + [
                'Tooth types group matching positions across quadrants.']
        if mode in ('tooth-sam2', 'tooth-types'):
            from transformers import Sam2Model, Sam2Processor

            self.mask_processor = Sam2Processor.from_pretrained(model_directory / 'sam2', local_files_only=True)
            self.mask_model = GroundedSegmenter._load(self, Sam2Model, model_directory / 'sam2')
            self.provenance['sam2'] = json.loads((model_directory / 'sam2/provenance.json').read_text())
            self.model_name += ' + SAM 2.1'

    @staticmethod
    def draw_label(frame, coordinates, text, color):
        left, top, right, bottom = map(int, coordinates)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        origin = (left, max(14, top - 4))
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)

    def cavity_candidates(self, image, rows, boxes, frame):
        if not rows:
            return rows, frame
        self.observation_counts['candidates_before_tooth_filter'] += len(rows)
        detection = self.tooth_locator.detect(image, 'teeth.', 0.5)
        tooth_boxes = [box.detach().cpu().tolist() for box, label in
                       zip(detection['boxes'], detection['text_labels']) if 'teeth' in label.lower()]
        retained = []
        for row, box in zip(rows, boxes):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            overlaps = (max(0, min(x2, tx2) - max(x1, tx1)) * max(0, min(y2, ty2) - max(y1, ty1))
                        for tx1, ty1, tx2, ty2 in tooth_boxes)
            if max(overlaps, default=0) < area * 0.5:
                self.observation_counts['rejected_outside_tooth_regions'] += 1
                continue
            row['category'] = 'cavity_candidate'
            self.draw_label(frame, box, 'Possible cavity', (70, 185, 255))
            retained.append(row)
        return retained, frame

    def predict(self, source_frame):
        height, width = source_frame.shape[:2]
        scale = min(1.0, 960 / max(width, height))
        frame = cv2.resize(source_frame, (round(width * scale), round(height * scale)))
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prediction = self.detector.predict(frame, imgsz=self.image_size, conf=self.confidence,
                                          iou=0.45, agnostic_nms=True, max_det=32, device=self.device,
                                          verbose=False)[0]
        boxes, rows = [], []
        for box in prediction.boxes:
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
            x1, x2 = np.clip([x1, x2], 0, image.width).tolist()
            y1, y2 = np.clip([y1, y2], 0, image.height).tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(box.cls.item())
            if self.mode == 'cavity-review' and class_id != 0:
                continue
            boxes.append([x1, y1, x2, y2])
            rows.append({'model_class_id': class_id, 'model_label': self.names[class_id],
                         'confidence': float(box.conf.item()),
                         'bbox_xyxy_normalized': [x1/image.width, y1/image.height,
                                                  x2/image.width, y2/image.height]})
        if self.mode == 'cavity-review':
            return self.cavity_candidates(image, rows, boxes, frame)
        if self.mode == 'tooth-fdi':
            strongest = {}
            for row, box in zip(rows, boxes):
                number = row['model_label']
                if number not in strongest or row['confidence'] > strongest[number][0]['confidence']:
                    strongest[number] = row, box
            for number, (row, box) in strongest.items():
                row.update(fdi_number=number, tooth_name=fdi_name(number))
                self.draw_label(frame, box, f'FDI {number}?', (70, 220, 110))
            return [row for row, _ in strongest.values()], frame
        if not boxes:
            return [], frame
        inputs = self.mask_processor(images=image, input_boxes=[boxes], return_tensors='pt').to(self.device)
        with torch.inference_mode():
            segmented = self.mask_model(**inputs, multimask_output=False)
        masks = self.mask_processor.post_process_masks(
            segmented.pred_masks.cpu(), inputs['original_sizes'].cpu())[0]
        qualities = segmented.iou_scores.detach().cpu().reshape(-1).tolist()
        retained = []
        for row, mask, quality in zip(rows, masks, qualities):
            binary = mask.squeeze().cpu().numpy().astype(np.uint8)
            if not binary.any() or not np.isfinite(quality):
                continue
            row.update(mask=binary, mask_quality_estimate=quality)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (70, 220, 110), 1)
            if self.mode == 'tooth-types':
                x, y = row['bbox_xyxy_normalized'][:2]
                origin = (int(x * image.width), max(14, int(y * image.height) - 4))
                text = row['model_label']
                cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3)
                cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            retained.append(row)
        return retained, frame
