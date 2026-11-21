# DentiLens XR 4.6 real-world validation

## Acquisition and annotation

The [acquisition registry](../datasets/video_sources) lists ten local
recording leads. Each is pending, quarantined, and ineligible for
annotation, training or evaluation as a benchmark source. No permission
for model use was inferred from downloading or prior demonstrations.
The Getty preview remains quarantined. The DentalGuard numbered images
remain an archive benchmark, without complete source videos, time-aligned
tracks or patient identity clearance. No newly authorized real dental
recording was available.

The counts for eligible dental footage, eligible recording groups and
complete reviewed dental tracks are all zero. Dental-video duration,
resolution and class statistics are therefore unavailable for the new
benchmark. Permission checks reported all ten leads excluded. The schema
separates eight allowed purposes, requires explicit consent for potential
identification, and rejects recorded expiry or inactive status. Controlled
media stay local. Registration records the checksum and metadata, and
extraction or scoring rechecks the checksum. Source- and patient-group
conflicts are rejected when metadata is present. Exact duplicate video
bytes are rejected; edited copies still require source review.

The new [annotation workflow](../datasets/annotations/README.md) imports
CVAT native-frame tracks, normalizes original-resolution boxes, preserves
occlusion and track IDs, and records reviewed entry and observed exit
events. All scheduled samples, including empty ones, require a review
attestation bound to the source and XML hashes. Completely invisible tools
carry no box. A second independent export can be compared for missed
boxes, class and visibility disagreements. Use the `compare-annotations` command with `--second-xml` and
`--second-review` to generate the disagreement report. The existing Dataset Explorer
now shows source type, group, permission, split and annotation coverage,
with counts for reviewed tools, occlusion and empty frames.

## Detector resolution results

The following runs use the same 637 DentalGuard validation frames and
549 reference tools. At every resolution the cutoff is 0.15. There are no
locked-test resolution changes.

| Input | Matched | Precision | Recall | AP50 | mAP50–95 | Unmatched | Tool-overlap duplicates | Glove overlap | Small-tool matches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 30 | 1.78% | 5.46% | 0.62% | 0.29% | 1,651 | 1 | 39 | 0/12 |
| 640 | 37 | 1.40% | 6.74% | 0.40% | 0.14% | 2,606 | 1 | 94 | 0/12 |
| 768 | 36 | 1.31% | 6.56% | 0.40% | 0.11% | 2,713 | 0 | 140 | 0/12 |

Among the 1,062 additional 768 px unmatched proposals, the category
differences versus 512 were +868 low tool overlap, +114 annotated
non-instrument overlap, +81 partial tool overlap and −1 overlap with
an already matched tool. Tool labels cannot be wrong at this detector
stage because its output class is generic. The low-overlap group is
unverified background, given incomplete source annotation. Small tools
have a reference short edge under 16 source pixels. A ranked candidate can
overlap a missed reference even if its confidence is too low or it was
assigned elsewhere. The evidence supports keeping 512 px. Detailed
[resolution results](instrument-46-resolution-results.json) contain
per-class recall and separate unmatched categories. No new detector or
classifier training was performed without cleared expansion data.

## End-to-end runner and verification

The benchmark writes independent localization, manual-crop classification,
trajectory and named visibility results. IDF1, switches, fragmentation,
false tracks, first detection/name delays and missed sampled instrument-
seconds are derived from reviewed tracks. A synthetic decoder fixture
with six scheduled frames and one artificial annotation track exercised
released and calibrated shadow weights through the complete runner.
It produced both independent prediction histories and the separate
manual-crop classifier report. Fixture timings validate execution only,
and there are no real-dental full-video scores to report. The process RSS
and stage durations are included in the local runner output. In the 512 px
smoke run, peak process RSS was about 585 MB and detector time about 0.41 s.
A second smoke run exercised all three resolution comparison paths.
These measurements do not establish resolution-specific production
memory or latency.

The [baseline audit](dentilens-xr-46-baseline-audit.md) documents the
recomputed confidence measures and test history. Both old production
weights and the separate local shadow checkpoint were preserved. All 42 automated tests passed. Existing
mirror overlay, review queue and shadow panel passed their browser checks.
The Dataset Explorer inventory and group filter also passed browser checks.
No automatic recognition improvement or clinical validation is claimed.

## Reproduce

Run commands from the application repository root. Keep the working
registry and source/annotation paths private and under local storage.

```bash
D=runtime/intake/dentalguard-4.3
A="/path/to/DentalGuard-PGIE-Master.v3i.yolov8-obb.zip"
R=runtime/intake/instrument-46
mkdir -p "$R"
for size in 512 640 768; do
  .venv/bin/python scripts/evaluate_detector.py "$D" "$A" "$R/$size.json" \
    --split valid --image-size "$size" --errors "$R/$size-errors.json"
done
.venv/bin/python scripts/compare_detector_resolutions.py "$D" "$R" "$R/resolutions.json"
.venv/bin/python scripts/inspect_dataset.py "$D" "$A" "$R/explorer.html" \
  --video-registry datasets/video_sources
.venv/bin/python -m pytest -q
```

For each cleared recording, prepare the [source JSON](../datasets/video_sources/README.md),
[CVAT export and review JSON](../datasets/annotations/README.md), and optional
local calibrated shadow checkpoint directory. The check command returns
nonzero when no registered sources are eligible.

```bash
.venv/bin/python scripts/video_benchmark.py register source.json /local/registry/source.json
.venv/bin/python scripts/video_benchmark.py check /local/registry annotate
.venv/bin/python scripts/video_benchmark.py extract /local/registry/source.json /local/sampled-frames
.venv/bin/python scripts/video_benchmark.py validate /local/registry/source.json tracks.xml review.json /local/reviewed.json
.venv/bin/python scripts/video_benchmark.py evaluate /local/registry/source.json tracks.xml review.json /local/results \
  --registry /local/registry --shadow /local/shadow-checkpoint
```

Use `--sizes 512 640 768` on validation recordings for separate detector
geometry and classifier comparisons. The next dependency is appropriate
clearance for representative footage and complete, independent annotation.
