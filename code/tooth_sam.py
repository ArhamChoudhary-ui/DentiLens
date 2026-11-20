"""Use the locally fine-tuned tooth detector to prompt SAM 2.1."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from grounded_sam import GroundedSegmenter


class ToothSegmenter:
    model_name = 'DentiLens intraoral YOLOv8n + SAM 2.1 tiny'
    prompt = None
    names = {0: 'tooth'}
    limitations = ['Experimental video detector: supplied clips exposed non-tooth false positives. '
                   'Still-image detection mAP does not establish video or mask accuracy.']

    def __init__(self, weights, model_directory, confidence=0.35, device='cpu'):
        from ultralytics import YOLO
        from transformers import Sam2Model, Sam2Processor

        weights, model_directory = Path(weights), Path(model_directory)
        self.confidence, self.device = confidence, device
        metadata = json.loads(weights.with_suffix('.json').read_text())
        digest = hashlib.file_digest(weights.open('rb'), 'sha256').hexdigest()
        if digest != metadata['weights_sha256']:
            raise ValueError('Tooth detector weights do not match their provenance')
        self.detector = YOLO(str(weights))
        if self.detector.task != 'detect' or self.detector.names != self.names:
            raise ValueError('Expected a one-class intraoral tooth detector')
        self.mask_processor = Sam2Processor.from_pretrained(model_directory / 'sam2', local_files_only=True)
        self.mask_model = GroundedSegmenter._load(self, Sam2Model, model_directory / 'sam2')
        self.provenance = {'detector': metadata, 'sam2': json.loads(
            (model_directory / 'sam2/provenance.json').read_text())}

    def predict(self, source_frame):
        height, width = source_frame.shape[:2]
        scale = min(1.0, 960 / max(width, height))
        frame = cv2.resize(source_frame, (round(width * scale), round(height * scale)))
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prediction = self.detector.predict(frame, imgsz=416, conf=self.confidence,
                                          iou=0.45, max_det=32, device=self.device,
                                          verbose=False)[0]
        boxes, rows = [], []
        for box in prediction.boxes:
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
            x1, x2 = np.clip([x1, x2], 0, image.width).tolist()
            y1, y2 = np.clip([y1, y2], 0, image.height).tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            rows.append({'model_class_id': 0, 'model_label': 'tooth',
                         'confidence': float(box.conf.item()),
                         'bbox_xyxy_normalized': [x1/image.width, y1/image.height,
                                                  x2/image.width, y2/image.height]})
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
            retained.append(row)
        return retained, frame
