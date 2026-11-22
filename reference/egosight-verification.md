# EgoSight XR verification

Verified locally on 22 November 2026 using the Apple Silicon Mac CPU path. The browser calls FastAPI, which runs the independent Track 4 CLI in a worker. The server validates the result before it makes the original JSON available.

## Inference on organizer-listed videos

| File | Returned category | Timed actions | Highlights | Local validation |
| --- | --- | ---: | ---: | --- |
| `P02_13.mp4` | cooking (manifest category) | 43 | 1 | PASS |
| `RockClimbing_oVnAVcbMoSM.mp4` | rockclimbing | 0 | 3 | PASS |
| `Skate_el49s-RlIa4.mp4` | other | 0 | 1 | PASS |
| `Surfing_7bzMGVKaiCQ.mp4` | other | 0 | 2 | PASS |

The JSON download matched the runner output for all four clips. The FPVSum transfer model supplies activity categories and highlights. Its action list is empty. An upload of the climbing footage under an unlisted name also returned rockclimbing with three highlights and passed schema validation. This verified the unlisted-upload path with the same source footage.

## Browser, API and CLI

| Check | Result |
| --- | --- |
| Chromium upload, preview, processing state and model result | PASS |
| Timeline seek and highlight playback to the end of the selected interval | PASS |
| Open JSON, download and compare it with the runner file | PASS |
| Reopen saved history and open benchmark and model settings | PASS |
| Mobile overview at 390 pixels, desktop workflow at 1440 pixels | PASS |
| FastAPI health, settings, errors, malformed upload cleanup and concurrent-run rejection | PASS |
| Pytest and Vite production build | PASS (59 tests) |
| Staging ZIP integrity and repeat inference from its launcher | PASS (byte-for-byte P02_13 JSON match) |


## Samsung staging coverage

The staging ZIP contains 102 valid outputs out of 185 eligible IDs, including 55 EPIC-KITCHENS-100 and 47 FPVSum clips. The supplied manifest still lists all 187. The two approved unavailable scuba exclusions are recorded in [the organizer update](../submission/ORGANIZER_UPDATE.md). The 83 outstanding eligible videos and the official organizer score remain pending. The local checker verifies output shape, vocabulary and timing. Samsung’s weights are action coverage 40%, action mAP 25%, highlight F1 25% and label accuracy 10%.

The written highlight rules require a four-second segment and target at most 15% of source duration. For very short clips, the checker records a warning when those constraints conflict.

The standalone workspace was reverified after the repository cleanup.

## Why This Moment?

Chromium processed `P02_13.mp4` through upload, preview and inference.
The result had 43 timed actions and one highlight, 24.00–28.00 seconds.
The selected explanation showed the actual ranked window's “set alarm”
prediction at a 44% model score. The score is for the window prediction.
The selector also applies its heuristic, duration budget and overlap rules.
The targeted unit test passed for both a scored candidate and a fallback
with no confidence score. The browser download matched the runner file
and passed the team validator with zero warnings.

## Final reproducibility check

The browser download matched the packaged CLI output byte for byte on
`P02_13.mp4`. Both used the original TSN. The team validator checked all 102
staged outputs and reported zero errors and three short-clip budget warnings.
No complete local footage matching the other 83 eligible manifest IDs was
found in the project runtime or Downloads.

The published R3D-18 path still returns `other` for the tested skateboarding
and surfing footage. The surfing clip’s mapped surfing score led at 0.0583,
below the existing 0.10 fallback threshold. Skateboarding was outside the
top two mapped categories for the skate clip. EPIC calibration does not change
these results. The organizer footage and hidden FPVSum labels were excluded
from the calibration data.

## Frozen TSN calibration

Six complete official EPIC training videos were checksum-verified. Four
participants supplied 59 training windows (P02_116, P08_07, P06_109, P26_15).
Two other participants supplied 70 validation windows (P03_116, P04_103).
The CPU run took 46.9 seconds including frame extraction. The best validation
loss was at epoch 10.

| Held-out metric | Original TSN | With calibration |
| --- | ---: | ---: |
| Verb top-1 | 65.7% | 65.7% |
| Noun top-1 | 75.7% | 75.7% |
| Both correct | 55.7% | 55.7% |
| Combined cross-entropy loss | 1.7401 | 1.7242 |

The small run showed no top-1 gain. The app and staging ZIP use the original
pretrained model. The calibration participants were held out from fitting
the calibration layer. They may have been seen by the upstream TSN. The
reported values are window classification checks. The run did not measure
official Track 4 performance. [Configuration and learning curves](tsn-calibration-results.json).

## 11 PM readiness verification

The original checkpoints reproduced all four real-video results in the table.
Cooking is the organizer-supplied category for the EPIC path. On P02_13, the
EPIC TSN aggregates sampled logits into action windows. The highest-scoring
actions included `set alarm` at 25–27 s (87.1%), `open oven` at 19–21 s
(62.2%), and `close oven` at 23–25 s (61.1%). The distinct four-second
highlight window was 24–28 s and its selected action-pair score was 44.0%.

The FPVSum path uses R3D-18 and returns an empty action array directly.
The model sums softmax probabilities for Kinetics classes that map to an
organizer category within each four-second window. The full-video score
averages those mapped scores over the clip. On RockClimbing, the full-video
score was 18.7% and the first window score was 87% rounded. TSN verb/noun
mapping, calibration weights and action filtering are absent from this path.

The climbing highlights showed hands reaching or holding the rock at
0–4 s, 52–56 s and 68–72 s. The last two have a similar viewpoint.
The three intervals are sorted and disjoint, and retain 12 of 83.5 seconds
(14.4%). The selector's existing ranking was retained. The new UI identifies
the selected range, separates full-video and window scores, and identifies
a fallback without presenting it as recognition. It hides the empty action
track on activity-only results. Highlight playback now settles at the
selected endpoint. Browser checks passed for replacing a cooking clip with
climbing, matching both downloads to their backend JSON, and mobile width.

The local experimental calibration checkpoint loaded against the pinned TSN
SHA-256. Its top-1 accuracies remain tied. No training was repeated during
this verification.
