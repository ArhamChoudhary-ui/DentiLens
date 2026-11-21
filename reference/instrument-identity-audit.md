# Instrument identity audit

The video scan loads a checksum-verified single-class YOLOv8 detection
checkpoint at 512 px, matching the video-training input size. The raw YOLO
results contain frame-relative geometry, object score and the single class
ID 0, Dental instrument. The backend stores the score and normalized
polygon, and the prior UI renders a fixed “Experimental candidate” label
for ID prefixes from this detector. There is no lost subtype field.
In the bedside clip the 8.4084 s observation holds a box and a detector
score, with no subtype probability.

The separate 14-class YOLOv8-OBB checkpoint loads only for the still-photo
scan. Its verified model class mapping is used in that path. It detects
oriented boxes and assigns them labels at 320 px. It does not classify
arbitrary proposals from the video detector. Its earlier bedside scan
returned zero boxes from 63 sampled frames.

A separate crop classifier can run downstream of video detection while
preserving the original detector records, coordinates and scores.

## Reproduction

The two stages use separate scores. The existing detector proposes a
region at 512 px. The classifier crops it with 10% padding, converts it
to grayscale, preserves its aspect ratio, and scores the standardized
crop. Classification predictions are stored in a separate sidecar
keyed to the original observation ID. No box or detector score is
replaced. The browser shows names only when the class passes its
validation threshold and has repeated support on a spatial track.

The [Dental Instrument Detection V3](https://universe.roboflow.com/mechatronics-engineering/dental-instrument-detection-v3-yymto)
export by Mechatronics Engineering is CC BY 4.0. Use the original
archive and put the four local video files beside a copy of
`instrument-identity-frames.json`, using the file names in that manifest.
The crop splits preserve the export's photo splits after grouping
originals and removing byte-for-byte duplicates. Each reviewed video
belongs to one split.

```bash
python scripts/train_instrument_identity.py /path/to/V3.zip /path/to/manifest.json /path/to/run \
  --epochs 20 --foundation /path/to/mobilenet_v3_small-047dcff4.pth
```

Install the run's `instrument-identity.pt` and `instrument-identity.json`
in `models/`. A scan backed by the trained checkpoint can propose
identities. Existing geometric scans can be classified with
**Identify instruments**. The named suggestion, probability, box and
source time remain tied to the original detection. Only reviewer
decisions enter confirmed instrument exports.

## Measured results

A three-layer CNN trained from scratch for 12 epochs enabled zero names
at the validation cutoff. The second run trained a classification head
for 20 CPU epochs over frozen [MobileNetV3 Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)
features. Its trained checkpoint is about 6 MB. Its photo validation
selected thresholds for mirror, forceps, handpiece, orthodontic pliers
and wire cutter. All other classes remain uncertain after thresholding.
The pliers class had no positive test photos.

On 700 held-out photo crops, raw accuracy was 74.1%. Applying the
thresholds yielded macro-F1 0.343 and 87.0% uncertain results. On nine
reviewed video crops from two clips, raw accuracy was 2/9 and the
classifier abstained on all nine. This includes the three held-out
glove-region crops. The app's additional track consensus can further
reduce displayed names. Neither split establishes clinical accuracy.

The bedside scan processed 101 existing boxes, kept their original
scores and timestamps, and produced zero automatically named boxes.
The two reviewed labels remain available. The current data does not
establish automatic subtype reliability on unseen dental video. More
independent labeled video crops and negative examples are needed,
including distinct probe, scaler and suction examples.

See [class metrics and confusion matrices](instrument-identity-results.json).

Complete scans also produced zero accepted names on the reflection clip
(155 proposals) and braces clip (81 proposals). On the portrait training
source, the app suggested dental mirror on 148 of 320 proposals. That
count is repeated frame-level output from a known scene and does not
measure generalization.
