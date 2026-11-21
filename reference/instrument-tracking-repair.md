# Instrument tracking and context repair

## Baseline

Video proposals are sampled at five frames per second. The experimental
YOLO detector produces one generic instrument class. The two crop
classifiers attach instrument identity scores, then the spatial association
and track agreement rules determine which names the UI can show. A high
detector score is not a named identity. The prior generic "Uncertain
instrument" row combined several distinct cases.

On 6502638, the previous committed association created 21 tracks from
320 raw proposals, including 12 single-frame tracks. The saved
classification sidecar from an in-progress repair had 159 named rows,
154 type-uncertain rows and seven suppressed duplicate rows. The
recording also had one corrected reviewer label. On 5892118, the
previous association created 17 tracks from 101 proposals (six
single-frame); all 101 had uncertain type and two had saved review
labels. The corresponding braces counts were 24 tracks from 81
proposals (16 single-frame), zero AI names and one rejection.

Two mirror-sized boxes overlapped from 2.20 to 2.60 s. In the earlier
record, both mirror crops at 2.60 s scored about 99.98%, but both were
treated as single-frame tracks. An uncommitted implementation
already had configurable overlap consolidation and a short gap
window. Replaying the saved responses through that implementation
reduced the portrait sequence to seven tracks (two single-frame)
and put the mirror and explorer on different tracks at all three
timestamps. This established that the first repair was a tracking
change, without a change in the instrument models.

Counts describe proposals and tracks, not unique physical instruments.
Model weights and class maps are checksum-verified and the existing
raw boxes and probabilities were available. There was no evidence of
an incorrect mirror class mapping. Earlier data did not distinguish
hands or mouths from generic unknown crops at inference time.

## Implementation

The crop models now score each raw proposal before a single canonical
association pass. A duplicate needs compatible class evidence and either
IoU of at least 0.72 or at least 88% containment of the smaller box, with
a nearby center and substantial overlap of the larger box. Priority goes
to a confident class, then a tight representative. The pairwise check
avoids chaining overlap through unrelated objects. Raw boxes and every
class score are retained. The representative and its aliases have
separate IDs.

The association window is 0.43 seconds at a five-frame-per-second
sampling rate. A track can resume after a missed sample when geometry
supports a unique match in both directions. A stable prior identity
blocks an incompatible new class. Ambiguous matches end the old
association. The UI displays aggregate confidence from at least two
same-class supporting frames. It only reuses that name near supporting
evidence on the same track. Playback interpolates only between two
matched observations inside the gap window.

Results distinguish type uncertainty, unclassified candidates,
interrupted tracks and track termination after prolonged absence.
The review panel exposes the representative, frame history, gap and
termination information. Existing reviews use raw proposal IDs.
The confirmed export uses saved reviewer actions.

## Context model

The separate DentalGuard context classifier was trained on CPU for
22 epochs after verifying the mirror replay with the existing instrument
models. It scores gloved hands, bare hands, open mouths and other crops.
It uses the source-group split from the DentalGuard experiment and the
same local MobileNetV3 weights. The deployed checkpoint names are
`models/instrument-context.pt` and `models/instrument-context.json`.

Reproduce using the command in
[instrument-identity-dentalguard.md](instrument-identity-dentalguard.md),
with `--context` added. The full test comprises 419 crops from four
held-out numbered source groups. Raw top-class accuracy was 82.8%.
At the validation-selected cutoffs, held-out precision and recall were
87.9% and 72.5% for gloved hands, 94.1% and 70.3% for bare hands, and
86.7% and 87.5% for open mouths. These are crop scores. The proposal
detector still runs as an instrument localizer.

In 5892118, visual inspection found an actual glove and the shaft and
head of a mirror in the eight context-positive regions. The app
therefore displays "Gloved hand in crop" as context and keeps the
candidate reviewable. To suppress an actual false proposal, the
reviewer can use **Mark non-instrument**. The original AI evidence and
the review action are retained. The app decodes frames once per
classification pass and does no inference when opening a saved tab.

## Live replay and remaining uncertainty

The live API replay checked the raw-file checksum before and after
analysis. It also checked both physical tool tracks and the canonical
representative at the specified times. Playback was checked at the
sample times and at the times between samples.

| Source | Raw proposals | Duplicates | Tracks before → after | Single-frame tracks before → after | Named after | Type uncertain or unclassified after | Context after |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6502638 | 320 | 7 | 21 → 7 | 12 → 2 | 159 | 154 | 0 |
| 5892118 | 101 | 1 | 17 → 12 | 6 → 3 | 0 | 92 | 8 |
| Braces | 81 | 0 | 24 → 18 | 16 → 10 | 0 | 81 | 0 |

The following breakdown uses all 320 raw portrait proposals as the
denominator. Observation categories are disjoint. Track interruption
events use the track denominator separately.

| Cause or outcome | Count | Share | Evidence and resolution |
|---|---:|---:|---|
| Named on a stable track | 159 | 49.7% | Repeated matching class evidence |
| Unknown was the top raw class | 150 | 46.9% | No reliable instrument type came from the classifier |
| Below an enabled class cutoff | 3 | 0.9% | Kept uncertain |
| Class lacked an enabled cutoff | 1 | 0.3% | Kept uncertain |
| Duplicate proposal | 7 | 2.2% | Linked to its canonical box |
| Mapping/UI loss found | 0 | 0% | Mirror class and probabilities were present |

Four of the seven final portrait tracks had a sample gap or ambiguous
termination. The original problem around 2.60 s crossed several
one-frame tracks. The final mirror remains on V-4 through the replay;
the explorer remains on V-7. This scene does not validate automated
explorer naming on independent videos. There is no dense annotation
of every candidate to separate localization failures and unfamiliar
instruments within the unknown-class count.

| Source time | Mirror raw proposal IDs | Canonical mirror | Explorer canonical |
|---|---|---|---|
| 2.20 s | instrument-video-55-1, instrument-video-55-2 | instrument-video-55-2 | instrument-video-55-0 |
| 2.40 s | instrument-video-60-0, instrument-video-60-1 | instrument-video-60-0 | instrument-video-60-2 |
| 2.60 s | instrument-video-65-0, instrument-video-65-3 | instrument-video-65-0 | instrument-video-65-1 |

[Frame-level replay, model scores and source splits](instrument-tracking-repair-results.json).
