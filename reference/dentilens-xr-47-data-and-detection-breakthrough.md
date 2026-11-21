# DentiLens XR 4.7 data and detection

## Clearance and acquisition

[All ten recording leads](dentilens-xr-47-source-clearance-audit.md) remain
quarantined. Eligible real recordings, eligible groups and complete
reviewed real-dental tracks remain at zero. The [opportunity inventory](../datasets/video_sources/clearance/opportunities.json)
documents four source pages and the remaining rights checks. The local
packet contains ten asset drafts, four opportunity drafts, a template
and a private worksheet. No outreach was sent or footage downloaded.
The registry requires derived-data retention as a separate permission
before creating extraction, annotation or benchmark artifacts. The
Dataset Explorer exposes the eight individual purpose states, consent,
expiry, hash and annotation readiness without including patient identifiers.

## Detector experiments

Every row below used the same 637 archive validation frames, 549 reference
tool boxes, existing checkpoint, 0.15 operating cutoff and 0.001 ranking
floor. The locked test split was not used. Volumes are proposal counts,
at one-to-one IoU 0.50 for the operating results.

| Method | Matched | Precision | Recall | AP50 | mAP50–95 | Unmatched | Glove overlap | Tiny matched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full 512 | 30 | 1.78% | 5.46% | 0.62% | 0.29% | 1,651 | 39 | 0/12 |
| Full 640 | 37 | 1.40% | 6.74% | 0.40% | 0.14% | 2,606 | 94 | 0/12 |
| Full 768 | 36 | 1.31% | 6.56% | 0.40% | 0.11% | 2,713 | 140 | 0/12 |
| 512 full + fixed center ROI | 49 | 1.90% | 8.93% | 1.15% | 0.44% | 2,527 | 112 | 0/12 |
| 512 overlapping tiles | 61 | 1.69% | 11.11% | 0.59% | 0.19% | 3,548 | 178 | 0/12 |

The ROI variant combines the full image and its central 60% rectangle.
The tiles use four 65%-sized overlapping windows. Both preserve aspect
ratios through Ultralytics preprocessing, remap boxes into original
normalized coordinates, and use score-ranked IoU 0.50 NMS across
windows. The validation configurations retain separate pre-merge and
post-merge ledgers. Neither variant alters application production scans.

| Method | Low tool overlap, unverified | Partial tool overlap | Annotated non-instrument overlap | Already-matched tool overlap |
| --- | ---: | ---: | ---: | ---: |
| Full 512 | 1,496 | 109 | 45 | 1 |
| Fixed ROI | 2,168 | 221 | 136 | 2 |
| Tiles | 3,025 | 305 | 217 | 1 |

The categories use ground-truth geometry and do not equate weak overlap
with a verified background false positive. The source labels are not
exhaustive. The detector output is one generic tool class. Tile merging
removes spatial repeats without resolving visually ambiguous tools.

Measured CPU model processing on the Mac was 194.65 s and 477 MB peak
process RSS for full 512, 288.08 s and 509 MB for the ROI pass, and
322.46 s and 658 MB for tiles. Full 640 and 768 ran in 79.68 s and
92.89 s, with 615 and 639 MB peak respectively. Runs executed in
two separate concurrent batches with different contention. Compare
these as measured run observations. They cannot rank single-process
production latency across the two batches. CPU was used throughout.

**Decision.** Keep full-frame 512 in production. The fixed ROI has the
strongest AP trade-off of the variants and remains suitable for offline
shadow testing. None recovered the tiny cohort. No training or model
promotion took place.

## The 12 missed small-tool cases

[Case audit](instrument-47-small-cases.md) contains all 12 frame IDs,
measurements and visual findings. One source box measures exactly 16 px
to display precision and entered the original under-16 cohort because
its floating-point value was 15.999999999999943. It remains in the
12-case baseline for continuity. The strictly sub-16 cohort also had
zero matches on every method. Visibility and occlusion are unrecorded
in these frame annotations. Observations from the contact sheet flag
review priorities without relabeling source truth.

A fixed ROI produced one nominal match-sized box on a missed mirror
(IoU 0.517), with confidence near 0.001 and well below the 0.15 gate.
Most cases had poor overlap even before cross-window merging. The
pre-merge ledger identifies cases whose best-overlap candidate was
removed. This does not reveal candidates suppressed inside Ultralytics
prior to the recorded stage.

For supplied ground-truth crops, comparing 0%, 10%, 25% and 35% padding
gave the production general head 0/8 mirrors or explorers ranked
correctly at every scale. The production specialist ranked 2, 1, 0
and 0 correctly. The calibrated shadow head ranked 3/8 correctly at
all four scales. No named crop passed its respective model gate among
all 12 crops for any setting. Established encoding kept its aspect
ratio, and no image enhancement or arbitrary cutoff change was used.
The crop study is limited to this small validation cohort.

## Annotation, shadow and real-video readiness

Independent comparison now emits box IoU disagreements, class,
visibility and occlusion differences, unmatched boxes, track split/merge
and timing conflicts. Input copies remain separate. It reports evidence
for human adjudication. The benchmark records incorrect named sampled
seconds and per-track naming consistency, alongside localization, IDF1,
delays and missed instrument-seconds. Complete-video shadow reconstruction
writes native frame/time, ranked and pre-merge boxes, geometry and
separate released and experimental identity records.

A generated six-sample video exercised full, ROI and tile detector
geometry with the production and calibrated shadow classifiers. The
process peak was about 763 MB. This verifies execution only. No real
clinical video accuracy, tracking result or missed duration is available.
The last blocker is newly cleared, representative footage with complete,
independently reviewed track annotations.

## Reproduce

```bash
D=runtime/intake/dentalguard-4.3
A="/path/to/DentalGuard-PGIE-Master.v3i.yolov8-obb.zip"
R=runtime/intake/instrument-47
mkdir -p "$R"
.venv/bin/python scripts/audit_source_clearance.py datasets/video_sources "$R/local-packet"
.venv/bin/python scripts/inspect_dataset.py "$D" "$A" "$R/explorer.html" --video-registry datasets/video_sources
for method in full roi tiles; do
  .venv/bin/python scripts/evaluate_detector.py "$D" "$A" "$R/$method.json" --split valid \
    --method "$method" --timing --errors "$R/$method-errors.json" --predictions "$R/$method-predictions.json"
done
for size in 640 768; do
  .venv/bin/python scripts/evaluate_detector.py "$D" "$A" "$R/full-$size.json" --split valid \
    --image-size "$size" --timing --errors "$R/full-$size-errors.json"
done
.venv/bin/python scripts/compare_detector_resolutions.py "$D" "$R" "$R/comparison.json" \
  --runs full roi tiles full-640 full-768
.venv/bin/python scripts/analyze_missed_small_tools.py "$D" "$A" "$R/local-small-audit" \
  --runs "$R" --shadow runtime/intake/instrument-45-calibration
.venv/bin/python -m pytest -q
```

Use `video_benchmark.py evaluate --methods full tiles roi` on an
authorized validation recording. The existing gate requires source
approval for extraction, annotation, retention and evaluation. All 45 automated tests passed. Production mirror continuity, review queue,
live shadow comparison and clearance inventory passed browser checks. See
[structured detector results](instrument-47-detector-results.json).
