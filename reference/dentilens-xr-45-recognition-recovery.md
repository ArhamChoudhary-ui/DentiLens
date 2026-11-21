# DentiLens XR 4.5 recognition recovery

The classifier audit used the exact frozen 4.4 checkpoint, and the locked
1,281 crops reproduced 16/98 mirrors and 22/59 explorers at raw top one.
The existing production results remain 5/98 and 0/59 for the general head,
and 4/98 and 8/59 for the mirror specialist. The prior detector baseline is
26/526 tools at 512 px and 0/526 for the still-photo OBB model at its own
operating cutoff. See the [classifier audit](dentilens-xr-45-classifier-audit.md)
and [4.4 evaluation](dentilens-xr-44-model-recovery.md).

## Calibration and training

Validation-only temperature scaling reduced locked-test ECE from 0.1885
to 0.0644 and Brier score from 0.3550 to 0.3072. Top-one ordering stayed
the same. The 95% gate accepted three correct explorers on test, one
incorrect explorer on validation, and zero mirrors on either split.
At the diagnostic 90% line, test accepted 11 names at 45.5% precision.
Neither class qualified for automatic naming.

A controlled training-only exposure experiment reused the 81 screened
originals, with two mild brightness/contrast variants each and no new
source material. It used an RGB MobileNetV3 Small frozen backbone, an
AdamW head, seed 44, batch 24, and early stopping. It ranked 42/98 mirrors
and 13/59 explorers first. Validation accuracy was 72.1%, compared with
74.7% for the original head. On the locked set it accepted two names at
95%, with one correct. It was not selected for deployment or shadow use.
The [structured results](instrument-45-results.json) contain loss curves,
confusion matrices, confidence bins, per-class metrics and coverage curves.
The [intake workflow](instrument-45-intake.md) records review, rights,
source split and fingerprints. No newly cleared footage or periodontal
probe data was available. Exposure variants do not count as new recordings.

## Detector recovery

At both resolutions the evaluator used the same four validation recording
groups, the existing checkpoint, 0.15 score cutoff, normalized source
geometry, and one-to-one IoU matching. The locked test was not used to
choose a resolution.

| Input | Matched | Mirror | Explorer | Unmatched proposals | AP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 px | 30/549 | 3/162 | 1/81 | 1,651 | 0.62% | 0.29% |
| 768 px | 36/549 | 9/162 | 3/81 | 2,713 | 0.40% | 0.11% |

Glove-overlapping unmatched proposals rose from 39 to 140. The net
trade-off did not justify altering production. Annotations may be
incomplete, so unmatched counts are benchmark counts. Earlier AABB/OBB
conversion and checksum checks found no confirmed evaluator bug. Full-frame
training remains blocked pending appropriate clearance for identifiable
footage. Complete authorized video and time-aligned annotations are still
needed to benchmark recognition duration and track continuity.

## Shadow comparison

The optional shadow endpoint reads the saved production predictions and
runs a hash-checked experimental checkpoint over the same video boxes.
It writes a separate sidecar that never enters exports or reviewer labels.
The Instruments model panel shows checkpoint metadata, gate values and
runtime. Selecting an observation exposes both predictions and the reviewer
state. One local Mac run scored 320 saved proposals in 6.82 seconds,
with 158 disagreements and zero accepted shadow names. This clip has no
exhaustive reference labels, and disagreement does not measure accuracy.

Enable shadow comparison by setting `DENTILENS_INSTRUMENT_SHADOW` to a
local directory containing the generated `model.json` and matching
`model.pt`. Unset the variable and restart to disable it. Production
checkpoints remain in place, so a production checkpoint rollback was
unnecessary. No weights or examination media are published.

## Reproduce

```bash
D=runtime/intake/dentalguard-4.3
A="/path/to/DentalGuard-PGIE-Master.v3i.yolov8-obb.zip"
E=runtime/intake/instrument-44-experiment
C=runtime/intake/instrument-45-calibration
.venv/bin/python scripts/calibrate_instrument_recovery.py "$D" "$A" "$E" "$C"
.venv/bin/python scripts/train_instrument_recovery.py "$D" "$A" runtime/intake/instrument-45-exposure \
  --approvals reference/instrument-44-reviewed-crops.json --augment
.venv/bin/python scripts/evaluate_detector.py "$D" "$A" /tmp/validation-512.json --split valid
.venv/bin/python scripts/evaluate_detector.py "$D" "$A" /tmp/validation-768.json --split valid --image-size 768
cp "$E/best.pt" "$C/model.pt"
DENTILENS_INSTRUMENT_SHADOW="$PWD/$C" bash run_full.sh
.venv/bin/python -m pytest -q
npm --prefix frontend run build
```

All 39 tests passed. The production build and live checks covered mirror
continuity, glove context, the unidentified queue, the saved shadow panel
and hiding comparison while retaining production output.
