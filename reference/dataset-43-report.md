# DentiLens XR 4.3 dataset and benchmark

## Sources and scope

- **DentalGuard-PGIE-Master v3** supplied 8,103 exported images and YOLO OBB labels. The local archive notice declares CC BY 4.0, which permits reuse and commercial adaptation subject to its terms, including attribution. The benchmark uses its 22 numbered recording groups and retains original source labels, file checksums, and coordinates. Patient consent, patient IDs, timestamps, source FPS, and complete recordings are unavailable. The archive remains local. Production training requires a separate provenance and privacy review.
- **Dental Instrument Detection V3 v3** supplied 5,213 640 × 640 images and YOLO OBB labels. Its local notice also declares CC BY 4.0. The original photo baseline was reproduced from this archive. It has 2,153 filename groups, augmentation variants, and no reliable patient or recording identifiers, so it is outside the source-grouped benchmark.
- **DentiLens local clips and mirror probes** remain separate from training and the 4.3 test split. The reviewed probe contains three mirror boxes from one clip and lacks exhaustive tool annotations. The two new clips do not provide reliable video-wide ground truth. No additional education or consented recordings were verified for this release.

See the [source registry](../configs/dataset_sources.yaml) and [canonical taxonomy](../configs/instrument_taxonomy.yaml). Explorer and periodontal probe have separate IDs. Generic source labels such as “composite instrument” remain unknown. Explicit context labels remain distinct from unclear instrument identities and other tools. The app uses the shared canonical display names for model suggestions.

## Dataset build

Both manifests reference the same source recordings but serve separate tasks. The detector receives whole frames, and the classifier receives boxes supplied by the source annotations. They share recording split assignments to prevent adjacent frames crossing folds.

| Split | Recording IDs | Frame images | Source boxes | Classification crops |
| --- | ---: | ---: | ---: | ---: |
| Train | 14 | 2,137 | 9,152 | 5,916 |
| Validation | 4 | 637 | 2,643 | 1,736 |
| Test | 4 | 573 | 2,500 | 1,608 |
| Total | 22 | 3,347 | 14,295 | 9,260 |

All retained images are 640 × 640. Totals by class across splits are below. Box counts include annotated context regions. Only instrument boxes count as detector targets.

| Canonical label | Frame boxes | Classification crops |
| --- | ---: | ---: |
| Dental mirror | 683 | 683 |
| Dental explorer | 341 | 341 |
| Dental forceps | 174 | 174 |
| Dental handpiece | 539 | 539 |
| Other instrument | 1,001 | 1,001 |
| Unknown instrument type | 243 | 243 |
| Non-instrument | 11,314 | 6,279 |

Excluded 4,705 files without verified recording split membership, 22 numbered frames with empty labels, 11 with duplicate or conflicting geometry, and 18 flagged for cross-split perceptual similarity. The crop build excluded 5,070 padded context crops that overlapped an annotated instrument before perceptual quarantine. The final splits show zero recording conflicts, exact-image matches, or flagged perceptual pairs at Hamming distance ≤4 on 64-bit dHash. Patient-level leakage cannot be checked on this source. The fingerprint screen reduces risk and cannot prove the absence of edited or related footage.

The Explorer shows split totals, class counts, source notes, and representative annotations with filters. Frame previews show box geometry without the source image. Tool crops are shown locally. Occlusion, quality ratings, exhaustive negative annotations, and an independent second-pass annotation review remain unavailable. The annotation guide supports collecting those fields for future recordings.

## Independent evaluations

The four original top-one baselines reproduced. The general model scored **74.1%** on V3 photo test crops and **39.1%** on the original sampled DentalGuard test crops. The specialist scored **81.0%** and **45.5%** respectively. Those baseline scores used the original native-class scoring and crop sampling.

### Detection

The released class-agnostic proposal model, at its current 0.15 score cutoff, matched **26 of 526** annotated instrument boxes on 573 test frames. Precision was **1.59%**, recall **4.94%**, AP50 **0.88%**, and mAP50–95 **0.38%**. Mirror localization was 1/98 and explorer localization was 2/59 at IoU 0.50. The evaluator counted 1,612 unmatched proposals, 22 of which overlapped a labeled glove, two a bare hand, and four an open-mouth region at IoU 0.50. Unmatched proposals are measured against the available annotations. The archive does not establish that every tool is annotated or that all 211 context-only test frames are tool-free. Class-specific recall is stratified localization recall, since the detector outputs a single instrument class.

### Classification

Both checkpoints saw the same 1,281 supplied test crops, with 98 mirrors, 59 explorers, 1,082 context negatives, and 42 ambiguous tool crops. Scores use a common mirror/explorer/unknown task. Other outputs collapse to unknown for the raw comparison only. The original per-head gate decisions and exact accepted labels are retained for coverage and false acceptance.

| Head | Raw top one | Raw top three | Macro-F1 | Rejected | False acceptance on negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| General | 84.54% | 94.30% | 0.328 | 96.57% | 37/1,082 |
| Mirror specialist | 83.53% | 100% | 0.378 | 99.77% | 3/1,082 |

The majority negative class drives the overall accuracy. The general model ranked mirror first on 5/98 and explorer on 0/59. The specialist ranked mirror first on 4/98 and explorer on 8/59. At deployed gates, the general model accepted 44 crops with one exact match, while the specialist accepted three crops with zero exact matches. The specialist's top-three result is automatic because it has only three classes. These scores are on a larger, negative-heavy split and cannot be compared directly with the 4.2 sampled baseline.

Complete recordings with time-aligned track annotations are still needed to measure missed duration, track-level consistency, duplicate tracks, and false tracks. No recognition improvement is claimed. Model weights, thresholds, and tracking stayed in place.

See [raw benchmark results](dataset-43-results.json) for the split IDs, per-class metrics, confusion matrices, calibration and coverage curves.

## Reproduce on an Apple Silicon Mac

From the repository root, use the existing Python environment with the local archive and choose a new output directory for each build.

```bash
ARCHIVE="/path/to/DentalGuard-PGIE-Master.v3i.yolov8-obb.zip"
OUT=runtime/intake/dentalguard-4.3
.venv/bin/python scripts/build_dataset.py "$ARCHIVE" "$OUT"
.venv/bin/python scripts/validate_annotations.py "$OUT"
.venv/bin/python scripts/check_data_leakage.py "$OUT"
.venv/bin/python scripts/evaluate_detector.py "$OUT" "$ARCHIVE" "$OUT/detector.json"
.venv/bin/python scripts/evaluate_classifier.py "$OUT" "$ARCHIVE" "$OUT/classifier.json"
.venv/bin/python scripts/inspect_dataset.py "$OUT" "$ARCHIVE" "$OUT/explorer.html"
.venv/bin/python -m pytest -q
```

The benchmark runs CPU inference with four PyTorch threads and reads media on demand from the archive. Re-run the 4.2 audit recipe for native photo and sampled DentalGuard baselines. Generated images and archive contents are excluded from Git.
