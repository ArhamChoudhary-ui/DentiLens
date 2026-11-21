# DentiLens XR submission readiness

Samsung XR Hackathon 2026 Track 4 · Action Intelligence Engine

| Requirement | Status | Evidence |
| --- | --- | --- |
| Project code | Complete for offline prototype | [Track 4 source and commands](../track4/README.md) |
| Offline inference | Verified | Eight organizer-listed clips. The extracted archive reproduced P02_13 byte for byte with the two-argument command. |
| Action prediction and localization | Verified on five kitchen clips | 157 scored verb–noun intervals with model-estimated boundaries |
| Highlight generation | Verified | Original-footage MP4 and chronological JSON intervals |
| Manifest coverage | 8 of 187 | [Per-video audit](../results/final_manifest_audit.json) |
| Local validation | 8 pass, 3 budget warnings | [Validation report](final-output-validation-report.md) |
| Official validation | Unavailable | Received student kits omit the organizer validator |
| Demo | Staged locally | 59-second walkthrough built from a real run. Public sharing of source footage remains pending. |
| Presentation | Staged locally as PPTX and PDF | Nine organizer-template slides. [Public copy](submission/DentiLens-Track4-Presentation.pptx) omits student IDs. |
| Public GitHub | Track 4 branch | [xr-4-video-intelligence](https://github.com/ArhamChoudhary-ui/DentiLens/tree/xr-4-video-intelligence) |
| Team identity | Pending | [Team metadata](../submission/TEAM_METADATA.md) |
| ZIP package | Private staging ZIP verified. Final ZIP pending. | Archive CRC and extraction passed. Source, two verified checkpoints and eight JSON files included. |
| Google Form | Pending | [Submission checklist](../submission/FINAL_SUBMISSION_CHECKLIST.md) |
| Automated tests | 99 passed | `python -m pytest -q` includes one added batch-resume regression test. |

## FPV category investigation

The skate clip visibly shows a skateboard. Its public annotation is
`skate`, an allowed label, and the R3D-18 transfer maps it to the Kinetics
`skateboarding` index in the checkpoint’s own category order. Inference
loads verified R3D-18 weights and uses Torchvision’s Kinetics transform.
The pooled skate probability was 0.00335. The strongest pooled mapped
category was rock climbing at 0.01614. Both fall below the 0.10 cutoff, so
the pipeline returned `other`. The result indicates a transfer-model miss.
The threshold and predictions were preserved.

| Public FPV label | Model output | Local highlight F1 |
| --- | --- | ---: |
| skate | other | 0.035 |
| rockclimbing | rockclimbing | 0.266 |
| surfing | other | 0.080 |

Category accuracy was 1 of 3 and mean local highlight F1 was 0.127 on
this subset. On five EPIC videos the local evaluator scored action mAP
0.497 at tIoU 0.1 and 0.464 at tIoU 0.5, with 2 of 7 per-video distinct
action pairs covered under the diagnostic’s half-action rule. These
results use public annotations only after inference. The subset cannot
establish full-manifest or organizer scores.

See the [technical report](dentilens-track4-final-report.md) for model
details and the [organizer draft](../submission/ORGANIZER_EMAIL_DRAFT.md)
for the remaining questions.
