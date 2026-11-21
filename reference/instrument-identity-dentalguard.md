# DentalGuard video-domain experiment

DentalGuard-PGIE-Master v3 combines still photos and numbered frames
from dental procedures. The export includes 8,103 images, including
347 explorer and 689 mirror annotations in numbered frames. The
publisher lists [CC BY 4.0](https://universe.roboflow.com/s-workspace-w5cu4/dentalguard-pgie-master).

All 22 recognizable numbered source groups were in the export's
training folder. We sampled labeled crops after assigning the whole
groups to 14 training, four validation, and four test groups.
No numbered group crosses the new splits. The filename groups identify
source recordings. Patient identity and recording independence beyond
those IDs are unverified.

The new model uses numbered frames and the previous reviewed local
training crops and V3 photo samples. It treats explicit hand and
open-mouth annotations as hard negatives where the box does not cover
more than 10% of any annotated tool. A visual audit found instruments
at small sizes, glove context in positive boxes and occasional baked-in
overlays. Box labels can be noisy. No raw frames are committed.

The training recipe retains the RGB MobileNetV3 crop encoder and
selects per-class confidence cutoffs using the four DentalGuard
validation groups. It keeps a cutoff only when validation yields at
least five accepted predictions at 90% precision or above.

```bash
python scripts/train_instrument_identity.py /path/to/V3.zip /path/to/manifest.json /path/to/run \
  --epochs 22 --foundation /path/to/mobilenet_v3_small-047dcff4.pth \
  --adapt-video --mirror-explorer --dentalguard /path/to/DentalGuard-PGIE-Master.v3i.yolov8-obb.zip
```

## Results

The 22-epoch run held out 293 validation crops and 253 test crops from
different numbered source groups. No byte-identical frames appeared
across numbered groups. The test included 59 explorer boxes, 82 mirror
boxes, and 112 negative crops. Raw crop accuracy was 67.2% on validation
and 69.2% on test. Neither class met the minimum precision and support
requirement at the candidate cutoffs, so no automatic names were
enabled. This checkpoint is a local experiment. The deployed
classifier was kept in place.

[Evaluation and source-group assignments](instrument-identity-dentalguard-results.json).
