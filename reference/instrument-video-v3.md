# Instrument refinement with video and V3 photos

The task is to locate handheld dental instruments in video. The checkpoint
outputs a single generic class. It does not assign a specific tool name.

## Data

- Mechatronics Engineering, [Dental Instrument Detection V3, version 3](https://universe.roboflow.com/mechatronics-engineering/dental-instrument-detection-v3-yymto),
  CC BY 4.0. The source archive SHA-256 is
  `51476382e4fa4621cfa4377086629c75cb1d46a9fe03782fcb2417c4fb5b83d4`.
  The recipe uses 214 photo groups from the exported training split, selected
  through the two mirror labels and Explorer No. 5. It converts oriented
  annotations to axis-aligned boxes and excludes dishes, cups, and mouth props
  from the handheld instrument category.
- User-supplied clip 6502638 adds 21 reviewed frames, exported both full-size
  and as closer crops. The labels identify the mirror head and the visible
  second instrument. Its subtype is unverified.
- The portrait and clip 6502766 appear to show the same exam. The portrait
  is used in training. Clip 6502766 is excluded from validation and test.
- Clip 33337556 supplies validation frames. The braces clip supplies the
  held-out box test. [Clip 6529138](https://www.pexels.com/video/dental-treatment-6529138/)
  supplies a separate reflection-only stress check. The test clips were not used
  for fitting. The braces benchmark was reused from the prior iteration.
- The 428 training image files have 384 unique exact-image hashes. No exact
  image hash is shared across training, validation and test.

## Reproduce locally

Use the prepared XR 4 pilot dataset and the last stable pilot checkpoint.
The source videos, photo archive and model weights are local inputs.

```bash
python scripts/refine_video_instruments.py /path/to/previous/data \
    /path/to/6502638.mp4 /path/to/previous/best.pt runtime/instrument-v3/grouped \
    --epochs 10 --archive /path/to/Dental-Instrument-Detection-V3.zip

python scripts/evaluate_video_instruments.py /path/to/source-folders \
    /path/to/5892118.mp4 models/dental-instruments-obb.pt \
    --candidate runtime/instrument-v3/grouped/fit/weights/best.pt \
    --portrait-split --portrait-video /path/to/6502638.mp4 \
    --reflection-video /path/to/6529138.mp4
```

Sparse test frames use axis-aligned IoU ≥ 0.50. The candidate score
threshold is selected on the validation clip by F1. The reflection check
counts boxes on 30 sampled frames, not a full-video false-positive rate.
The portrait score is a training-set diagnostic only.

## Results

The final run completed 10 CPU epochs at 512 px. The candidate threshold
selected on clip 33337556 was 0.15. The legacy detector used its existing
0.40 threshold. Each cell reports matches / extra boxes / misses.

| Sample | Legacy detector | New candidate |
| --- | ---: | ---: |
| Validation, clip 33337556 | 0 / 5 / 14 | 7 / 0 / 7 |
| Held-out braces | 0 / 4 / 7 | 0 / 3 / 7 |
| Portrait training clip 6502638 | 0 / 6 / 41 | 40 / 3 / 1 |

The reflection stress sample produced one legacy box and 31 candidate
boxes across 30 frames. Candidate braces AP50 was 0.049. Strong training
performance on the portrait scene did not carry over to the braces test
or reflection footage. The candidate remains a local experiment.
The live scanner continues to use the prior detector.

See the [recorded results](instrument-video-v3-results.json) for the model
checksum, thresholds, and per-source counts.

## Review the experimental model in the app

Copy the evaluated checkpoint to `models/video-instruments.pt` and choose
**Scan with experimental video model** in **Instruments**. The app verifies
the checkpoint against the recorded checksum. The resulting boxes are
unreviewed and use a generic instrument label. The scan has its own
saved results, so existing instrument reviews stay attached to their evidence.
