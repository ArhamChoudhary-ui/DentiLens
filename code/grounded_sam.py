"""Text-grounded region segmentation using published Grounding DINO and SAM 2 weights."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch


class GroundedSegmenter:
    def __init__(self, model_directory, confidence=0.5, device='cpu', include_mirror=False):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Processor, Sam2Model

        root = Path(model_directory)
        self.device = device
        self.confidence = confidence
        self.prompt = 'teeth.'
        self.names = {0: 'teeth'}
        self.include_mirror = include_mirror
        if include_mirror:
            self.names[1] = 'dental mirror head'
            self.model_name = 'Grounding DINO + SAM 2.1 (teeth and experimental mirror)'
            self.limitations = ['Experimental mirror-head localization can miss mirrors or confuse other objects with mirrors.']
        self.detector_processor = AutoProcessor.from_pretrained(root / 'dino', local_files_only=True)
        self.detector = self._load(AutoModelForZeroShotObjectDetection, root / 'dino')
        self.mask_processor = Sam2Processor.from_pretrained(root / 'sam2', local_files_only=True)
        self.mask_model = self._load(Sam2Model, root / 'sam2')
        self.provenance = {}
        for name in ('dino', 'sam2'):
            self.provenance[name] = json.loads((root / name / 'provenance.json').read_text())

        if include_mirror:
            self.provenance['mirror_detection'] = {
                'prompt': 'small round dental mirror.', 'confidence_threshold': 0.5,
                'box_width_to_height_range': [0.65, 1.55]}

    def _load(self, model_class, directory):
        model, information = model_class.from_pretrained(
            directory, local_files_only=True, use_safetensors=True, output_loading_info=True)
        if information['missing_keys'] or information.get('mismatched_keys'):
            raise ValueError(f'Incomplete or incompatible model weights in {directory}')
        return model.to(self.device).eval()

    def detect(self, image, prompt, threshold):
        inputs = self.detector_processor(images=image, text=prompt, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            prediction = self.detector(**inputs)
        return self.detector_processor.post_process_grounded_object_detection(
            prediction, inputs.input_ids, threshold=threshold, text_threshold=0.2,
            target_sizes=[image.size[::-1]])[0]

    def predict(self, source_frame):
        height, width = source_frame.shape[:2]
        scale = min(1.0, 960 / width)
        frame = cv2.resize(source_frame, (round(width * scale), round(height * scale)))
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection = self.detect(image, self.prompt, self.confidence)
        labels = detection['text_labels']
        boxes, rows = [], []
        for box, score, label in zip(detection['boxes'], detection['scores'], labels):
            if 'teeth' in label.lower():
                class_id = 0
            else:
                continue
            x1, y1, x2, y2 = box.detach().cpu().tolist()
            x1, x2 = np.clip([x1, x2], 0, image.width).tolist()
            y1, y2 = np.clip([y1, y2], 0, image.height).tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            rows.append({'model_class_id': class_id, 'model_label': self.names[class_id],
                         'confidence': float(score), 'text_prediction': label,
                         'bbox_xyxy_normalized': [x1/image.width, y1/image.height,
                                                  x2/image.width, y2/image.height]})
        if self.include_mirror:
            mirrors = self.detect(image, 'small round dental mirror.', 0.5)
            for box, score, label in zip(mirrors['boxes'], mirrors['scores'], mirrors['text_labels']):
                if 'mirror' not in label.lower():
                    continue
                x1, y1, x2, y2 = box.detach().cpu().tolist()
                x1, x2 = np.clip([x1, x2], 0, image.width).tolist()
                y1, y2 = np.clip([y1, y2], 0, image.height).tolist()
                if y2 <= y1 or not 0.65 <= (x2-x1)/(y2-y1) <= 1.55:
                    continue
                boxes.append([x1,y1,x2,y2])
                rows.append(dict(model_class_id=1, model_label=self.names[1],
                                 confidence=float(score), text_prediction=label,
                                 bbox_xyxy_normalized=[x1/image.width,y1/image.height,x2/image.width,y2/image.height]))
        if not boxes:
            return [], frame
        from torchvision.ops import batched_nms
        retained_indices = batched_nms(torch.tensor(boxes),
                                       torch.tensor([row['confidence'] for row in rows]),
                                       torch.tensor([row['model_class_id'] for row in rows]),
                                       iou_threshold=0.5).tolist()[:16]
        boxes = [boxes[index] for index in retained_indices]
        rows = [rows[index] for index in retained_indices]
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
            row['mask'] = binary
            row['mask_quality_estimate'] = quality
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = (245, 190, 70) if row['model_class_id'] == 1 else (70, 220, 110)
            cv2.drawContours(frame, contours, -1, color, 2)
            retained.append(row)
        return retained, frame
