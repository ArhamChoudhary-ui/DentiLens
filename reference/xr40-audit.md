# XR 4.0 instrument repair audit

Baseline branch at `23432c3`; the existing 23 tests passed. Runtime uses
YOLOv8n-OBB from the Roboflow v30 still-photo export, 320 px inference,
0.40 minimum prediction score and ByteTrack at about 5 Hz. The installed
classes and checkpoint checksum match the saved evaluation. Native
Ultralytics OBB suppression precedes tracking. The existing basename and
exact-image split checks removed original photo groups spanning splits.
Photo test mAP50 was 0.389. Patient identities and clinical evaluation were
unavailable.

The braces model produced 58 raw observations. At 12.00 seconds it assigned
Contra-angle handpiece to a right glove; other high-ranked observations
covered the same glove while the visible shaft entered from the left.
Persistence and changing how labels are shown did not solve these errors.
This is a localization/domain-shift problem. A separate localization task
will use true video frames, prioritize a broad instrument category, and
retain source-separated evaluation. Tool subtypes require additional
labeled clips.

The 5892118 bedside clip is a separate failure case. Validation compares
both models on visually annotated source frames and distinguishes visibly
annotated instruments from sampled frames where nothing could be verified.

## Video findings

The six local Pexels clips were evaluated on 44 hand-reviewed frames or
glove-only crops. Boxes approximate the visible part of an instrument. The
5892118 bedside and 5099636 cleaning clips plus glove crops from 4490363
were used for training. Threshold selection used 6502766. The independent
test contained 15 frames and 21 visible-part boxes from 6528841 (braces) and
33337556. Adjacent frames stay in their source split. The saved comparison
uses axis-aligned IoU ≥ 0.50 and matches boxes without requiring a subtype.

| Detector | Matches | Extra boxes | Misses | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy YOLOv8n-OBB at 0.40 | 0 | 9 | 21 | 0% | 0% |
| Video pilot YOLOv8n at 0.10 | 6 | 34 | 15 | 15.0% | 28.6% |

The threshold was selected on the validation split by F1. The stable pilot
used 18 CPU epochs and achieved 0.070 held-out AP50 and 0.034 AP50–95.
Its checkpoint SHA256 is
`d5ed080137894f4fa8674bd44053061f027e48fa4bf80917278fbf2c11f8f68a`.
The initial 12-epoch run and the grounding experiment failed to localize
tools consistently. None of the replacement checkpoints met the quality
needed for automatic review. A broader subtype evaluation, false positives
per minute and identity switch rates require dense video annotations.
The bedside clip was part of training, so performance on it is a diagnostic.

The application keeps still-photo candidates available for audit.
The 12.00 s glove prediction was dismissed in the local braces session.
The original candidate remains in its scan record. The reviewed-ROI tracker
uses OpenCV feature correspondences, forward–backward checks and RANSAC.
On the supplied clips it returned 18 nearby suggestions around the
21.64 s braces anchor and five around the bedside 12.51 s anchor. These
are inspectable matches, with the option to confirm individual frames.
The tracker can stop immediately during heavy occlusion. A reviewer
supplies the name and any tooth association.

See [frame manifest](xr4-instrument-frames.json) and
[saved comparison](xr40-instrument-evaluation.json).
The per-source figures include training and validation diagnostics and
should be read using their split labels. The repository stores the
annotations, evaluation, and reproduction scripts. Source clips and
weights are local.

## Audio and storyboard verification

Both supplied clips had no audio track. An independent synthetic speech
fixture processed by local Whisper returned “pause here, review the upper
front tooth” with source-relative timestamps. Storyboard generation
produced three chapters on the 27 s braces clip and one on the 12.61 s
bedside clip. Chapters reference source evidence and preserve reviewer
edits and the initial generated chapters. Clinical instruments and
anatomical relationships require a saved review.
