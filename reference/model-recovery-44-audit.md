# Instrument model recovery audit

## Runtime traces

The video scan samples the source at about 5 Hz. The installed single-class YOLOv8 detector uses Ultralytics aspect-preserving letterbox preprocessing at 512 px, runtime confidence 0.15, and Ultralytics NMS. Output boxes are decoded in normalized source coordinates and stored with the detector score, frame index, source time, and original class. The separate still-photo OBB detector uses 320 px and a 0.40 runtime cutoff. The benchmark reduces its four displayed OBB vertices to an enclosing AABB before matching. The 4.3 detector evaluator compares geometry only, with identical reference boxes and one-to-one score-ranked matching. AP uses ranked proposals above a 0.001 inference floor.

The crop pipeline uses 10% padding, image-bound clipping, aspect-preserving fit on a gray 96 px canvas, resize to 128 px, and ImageNet normalization. The general model uses grayscale broadcast to three channels during normalization and the mirror specialist uses RGB. Model loading compares the checkpoint hash and validates the output-class mapping against the metadata. The CPU scorer retains logits for offline diagnostics and stores the softmax probabilities and original prediction in the sidecar. Class gates and agreement across nearby observations determine which names reach playback. The browser renders the resulting model/review statuses and keeps reviewer labels separate from original scores.

No detector-coordinate, crop-coordinate, class-order, checkpoint-loading, or one-to-one matching bug was confirmed. The audit did confirm an evaluation interpretation limitation. A context-only DentalGuard frame does not establish that all objects in it are annotated. Count unmatched detections against available labels and avoid interpreting that count as a verified clinical false-positive rate. The three-class specialist's top-three result is automatic.

## Dataset integrity and representative quality

Both 4.3 benchmark outputs reproduced exactly. The frame leakage check again returned zero flagged split overlaps. The 18 quarantined frames remain excluded, and the 4.3 manifests were used read-only. Source IDs are the minimum grouping unit. Patient identifiers and split isolation at patient level remain unavailable.

| Class | Boxes train/validation/test | Recording groups train/validation/test | Test median short edge | Test short edge <16 px |
| --- | ---: | ---: | ---: | ---: |
| Mirror | 423/162/98 | 14/4/4 | 38 px | 13 |
| Explorer | 201/81/59 | 14/4/4 | 43 px | 10 |
| Periodontal probe | 0/0/0 | 0/0/0 | Unavailable | Unavailable |

A screen of two full frames from each training source group showed wide views, bright dental-light reflections, and faces or staff badges in some frames. The archive notice is CC BY 4.0. The notice does not establish consent or privacy clearance for identifiable subjects. No full-frame training was run. A 191-crop training-only shortlist exposed clipped tips, partial tools, occlusion, reflections, questionable labels, and source overlays baked into several crops. Only 81 tight crops passed a single visual screen for visible identifying context and acceptable class evidence. This is a limited local research subset. It has no independent second annotator. Source frames with identifiable content, new clips, and held-out crops were excluded from gradient updates.

The evidence supports two model-quality hypotheses. At least some instruments are undersized or poorly localized on whole frames. Crop-only recognition also remains vulnerable to specular glare, gloves, orientation, and source-label noise even when the tool is boxed. No finding establishes that swapping the network architecture alone would solve either problem.
