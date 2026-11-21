# DentiLens XR 4.4 model recovery

## Benchmark and decisions

The detector and classifier outputs reproduced the locked 4.3 results exactly. See the [pipeline and dataset audit](model-recovery-44-audit.md). The test data retains the original four recording groups, and all 18 frame quarantines remain in force.

| Detector | Threshold | Boxes matched | Mirror | Explorer | AP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing video YOLOv8 | 0.15 | 26/526 | 1/98 | 2/59 | 0.88% | 0.38% |
| Existing still YOLOv8 OBB | 0.40 | 0/526 | 0/98 | 0/59 | 0.039% | 0.0069% |

Both detector runs used the 573 identical test frames. The still model preserves its own runtime input size and score threshold and is evaluated offline by enclosing its oriented boxes in AABBs. The result did not justify switching detectors. Neither model was newly trained in 4.4. Identifiable training frames and incomplete background annotations prevent a representative full-frame training experiment on this export.

The detector error ledger records original archive member, source group, frame index, scores, coordinates, and best overlap. It lists 500 misses, 1,610 unmatched boxes, and two remaining boxes with an already matched annotated overlap. Among misses, 16/97 mirrors and 13/57 explorers had some ranked candidate at IoU ≥0.5, which includes candidates below the deployed cutoff and possible duplicate matches. On recording 102003 frame 16578, the missed mirror had a candidate at IoU 0.556 with score 0.0053. The missed explorer on frame 2142 had no candidate above IoU 0.264. This implicates both weak confidence and poor localization. Case records preserve reconstructible provenance without copying identifiable frames into the repository or adding unreviewed negatives to training.

| Crop classifier | Mirror top one | Explorer top one | Grouped macro-F1 | Rejected at frame gate |
| --- | ---: | ---: | ---: | ---: |
| Existing general MobileNetV3 | 5/98 | 0/59 | 0.328 | 96.57% |
| Existing RGB specialist | 4/98 | 8/59 | 0.378 | 99.77% |
| Experimental unified RGB head | 16/98 | 22/59 | 0.456 | 100% |

All three heads saw the same 1,281 reference crops. Grouped raw accuracy for the new head was 82.59%, top-three accuracy was 99.53%, and the ten-bin expected calibration error was 0.211. The dataset is dominated by context negatives. Native outputs distinguish mirror, explorer, non-instrument, and unknown. The experimental head called 95 of 1,082 true non-instrument crops mirror or explorer at raw top one. Neither class passed the predeclared 0.95 research gate on any test crop. The model is not eligible for promotion. No periodontal probe samples, standalone deployed explorer checkpoint, or complete annotated recordings were available for this comparison.

## Training run

A new classifier head was initialized over frozen RGB MobileNetV3 Small features from the existing specialist. The 81 [reviewed training crops](instrument-44-reviewed-crops.json) include 18 mirrors, 24 explorers, 28 gloved-hand negatives, and 11 ambiguous tool crops. Only the training split contributes gradients. The source-separated validation split selects the lowest weighted cross-entropy head, with square-root inverse frequency weighting, AdamW at 0.001, seed 44, batch 24, no augmentation, and early stopping after four stale epochs. The CPU run on the Mac stopped after seven epochs and selected epoch three. Training loss went from 1.327 to 0.275 and validation loss reached 0.813 at epoch three, then rose to 0.930 by epoch seven. Raw validation accuracy at the selected checkpoint was 74.7%. Training the frozen head took roughly 0.6 s after feature extraction. These timings do not measure live or full-video latency.

The approved, padded RGB crop byte hashes matched zero validation or test crops. A perceptual review screen found 11 validation and 23 test crops within four dHash bits of an approved crop. A tight crop can look alike even across different source frames, so those are unresolved similarity flags. The scored benchmark remains the original locked set. No flags were used to remove test examples, tune thresholds, or select a production model. Together with unavailable patient IDs and the small training subset, they limit the interpretation of the raw ranking improvement.

The experimental checkpoint and optimizer state are local and versioned with provenance and hashes. Production weights, thresholds, tracker, and review paths stayed in place. No in-app selector or rollback behavior was added because no experimental model passed the release gate. The detector CLI permits explicit offline model selection.

## Reproduce

```bash
D=runtime/intake/dentalguard-4.3
A="/path/to/DentalGuard-PGIE-Master.v3i.yolov8-obb.zip"
OUT=runtime/intake/instrument-44-experiment
.venv/bin/python scripts/check_data_leakage.py "$D"
.venv/bin/python scripts/evaluate_detector.py "$D" "$A" /tmp/dentilens-video.json --errors /tmp/dentilens-errors.json
.venv/bin/python scripts/evaluate_detector.py "$D" "$A" /tmp/dentilens-still.json --model still-obb
.venv/bin/python scripts/evaluate_classifier.py "$D" "$A" /tmp/dentilens-classifiers.json
.venv/bin/python scripts/train_instrument_recovery.py "$D" "$A" "$OUT" --approvals reference/instrument-44-reviewed-crops.json
.venv/bin/python -m pytest -q
```

Use a fresh output directory or add `--resume` with the same archived inputs, manifests, and approval list. The [structured results](instrument-44-results.json) include confidence and coverage, class-level metrics, both confusion matrices, the selected checkpoint hash, and episode losses. The [complete-recording protocol](instrument-44-recording-protocol.md) describes the annotation needed to measure missed duration and tracking. The next detector experiment needs cleared representative full frames, complete tool annotations, and reviewed negative frames. New training data should come from different source groups, and periodontal probe requires its own labeled examples.
