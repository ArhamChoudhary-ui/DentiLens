# Instrument identity v2

The miss in clip 6502638 at 1.40 s had two causes. The first classifier
scored the crop as unknown despite a visible mirror. The spatial matcher
also split the moving mirror: neighboring proposals still overlapped, but
their overlap fell below 0.45.

The updated association can match modest movement when the boxes still
overlap, have similar dimensions and have one match in each direction.
It retains frame-level boxes and separates ambiguous overlaps. The same
track IDs are used in review and playback. New association versions
invalidate cached classification results.

The training recipe adds verified cropped examples from the portrait
recording and a related landscape recording, including the tilted mirror
and angled explorer. Both recordings stay in training. It also adds
suction-tip examples from clip 5099636 and glove-only crops from 4490363.
The bedside and braces samples are excluded from classifier training.
The reflection clip is reserved for a separate full scan. The detector
checkpoint has its own training history, so bedside results alone cannot
establish end-to-end generalization.

Source archive: [Dental Instrument Detection V3](https://universe.roboflow.com/mechatronics-engineering/dental-instrument-detection-v3-yymto)
by Mechatronics Engineering, CC BY 4.0. The v2 recipe uses RGB crops,
retains frozen MobileNetV3 Small features and trains a new classifier head.
Video crops receive more sampling weight. Training-only augmentation
varies crop padding, horizontal orientation and brightness. Class families
that were grouped in v1 are split where the archive identifies distinct
tools, including artery forceps and distal end cutters.

The validation policy still requires a top score of at least 75% and
at least five qualifying predictions at 90% or higher precision before
enabling a class. Display also requires agreement on neighboring frames.
Classes observed only in training can remain uncertain.

## Reproduce

Put local source videos beside a copy of the
[frame manifest](instrument-identity-v2-frames.json) and run:

```bash
python scripts/train_instrument_identity.py /path/to/V3.zip /path/to/manifest.json /path/to/run \
  --epochs 22 --foundation /path/to/mobilenet_v3_small-047dcff4.pth --adapt-video

python scripts/check_instrument_identity_video.py --classifier /path/to/run /path/to/video.mp4
```

## Results and deployment

The broad head completed 22 epochs across 23 instrument types plus unknown.
It misclassified one held-out mirror crop as artery forceps at its
photo-selected operating point, so that checkpoint stays experimental.
Its per-class results and confusion matrices are included for comparison.

A second 22-epoch run focused on mirror, explorer and unknown. It retained
strong mirror scores on the tilted early portrait frames. Explorer did
not reach the precision gate on validation and stays uncertain for
automatic naming, including when it is the highest-scoring identity.
The reviewed explorer label still follows its supported track.

The app keeps its prior general classifier plus the new specialist. Each
checks its own per-class thresholds and requires track agreement. If
both accept different labels, the app marks the result uncertain. The
specialist is installed as `models/instrument-detail.pt` and
`models/instrument-detail.json`. It can be reproduced from the focused
training run by adding `--mirror-explorer` to the command above.
The metadata for the installed specialist, including its checkpoint
checksum, is recorded in the results JSON.

At the specialist's photo-validation floor of 90%, two open-mouth boxes
on the braces stress clip were labeled mirror. Deployment retained the
older 95% mirror floor. The braces replay at 95% is a regression check
after that decision. At deployment, the specialist produced no names on
81 braces proposals or 155 reflection proposals. It abstained on all
nine reviewed crops from the classifier-held-out video samples.

The updated app named the mirror at 1.00, 1.20 and 1.40 s on one spatial
track. The portrait training source had 159 named mirror proposals with
both models active, compared with 148 in the earlier app. The complete
5892118 replay still yielded zero automatic names among 101 proposals.
Existing reviewed labels were preserved. These results support a
specific improvement on the portrait scene. Additional independent
video of each instrument and hard negatives is needed to measure
generalization.

[Evaluation details and the app replay](instrument-identity-v2-results.json).
