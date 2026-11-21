# DentiLens XR Track 4 technical report

## Implementation

The dedicated offline entry point is `track4/run.sh input.mp4 output.json`.
No medical model participates in action prediction. The root dental launcher
and app are unchanged.

The EPIC route uses the published TSN RGB ResNet50 checkpoint trained on
EPIC-KITCHENS-100 training data. Inference checks SHA-256, processes source
frames at approximately 2 Hz, averages the RGB logits in two- and four-
second windows and scores train-observed verb–noun pairs. It retains
top-three alternatives outside the submitted JSON. Non-maximal suppression
removes overlapping proposals for the same pair. The model estimates window
boundaries and does not have a learned temporal boundary head.

The FPVSum route uses a Kinetics-400 R3D-18 model and a conservative mapping
to 11 FPVSum categories. It abstains to `other` when the pooled category
score is weak. Bikepolo, longboarding and plane lack direct mappings.
The FPV route emits no verb–noun actions as allowed by the guide. The
highlight selector combines category variety and confidence on that path.
Neither route verifies physical hand–object contact.

The selector operates on original source intervals, favors distinct pairs
with temporal diversity, and emits chronological non-overlapping clips.
The organizer JSON contains exactly the six documented fields. Model
metadata, examples and debug alternatives are kept outside official outputs.
Original video highlights are video-only FFmpeg re-encodes.

## Executed reference runs

| Manifest video | Dataset | Source length | Scored actions | Highlight | CPU analysis |
| --- | --- | ---: | ---: | ---: | ---: |
| P02_13 | EPIC-KITCHENS-100 | 29.51 s | 43 | 4.00 s (13.6%) | 7.38 s |
| P03_26 | EPIC-KITCHENS-100 | 11.13 s | 15 | 4.00 s (35.9%) | 3.23 s |
| Skate_el49s-RlIa4 | FPVSum | 37.24 s | 0 | 4.00 s (10.7%) | 6.29 s |

All three JSON files passed the local contract checker. P03_26's 4-second
minimum conflicts with the 15% budget, yielding the expected warning.
The other two stayed under budget. The rows represent actual reference
MP4s and CPU analysis time measured without export.

On the two EPIC videos, the public-label local evaluation found six
reference actions. Local action mAP was 0.6190, 0.6190, 0.5833, 0.5625,
and 0.5625 at tIoU 0.1 through 0.5. The local coverage diagnostic counted
one of four per-video distinct pairs using at least half of the reference
action duration. The single FPVSum sample received `other` for a reference
label of `skate`. Its local highlight F1 was 0.0348 on the public human score
array with a uniform time grid and deterministic tie ordering. These are
small-sample diagnostics. They do not reproduce the full official scoring
or show generalization.

Model inference uses only training class pair support or Kinetics category
mapping. Public validation labels were read separately for evaluation.
No model was trained or threshold tuned on these samples.

## Reproduction and release status

The local checker and packager use the supplied [organizer references](../track4/reference/student_package).
The packager refuses to build a final ZIP without a registered ID, all
187 checked JSON files and the actual organizer validator. The development
ZIP retains the required directory structure, hashes and local weights.
Do not redistribute model or video artifacts until the applicable terms
have been reviewed. Git excludes original benchmark clips and checkpoints.

```bash
cd track4
export DENTILENS_TEAM_ID=DEVELOPMENT
./run.sh "/local/media/P02_13.mp4" "/local/outputs/P02_13.json" \
   --device cpu --report "/local/review/P02_13-report.json" \
   --highlight-mp4 "/local/review/P02_13-highlight.mp4"
python scripts/check_submission.py --file "/local/outputs/P02_13.json"
python scripts/evaluate_local.py /local/EPIC_100_validation.csv /local/outputs/P02_13.json \
   --report /local/review/epic-local.json
python scripts/evaluate_fpv_local.py /local/gt_data.mat /local/outputs/Skate_el49s-RlIa4.json \
   --report /local/review/fpv-local.json
python scripts/package.py /local/outputs /local/Track4-DEVELOPMENT.zip --team-id DEVELOPMENT --development
```

All 98 automated tests passed. A local 8-slide PPTX/PDF and short
development walkthrough were produced. The optional in-app interface was
deferred per the P0 order. No official submission, combined benchmark
score, headset deployment or published demo URL exists. The missing inputs
are 184 manifest outputs, the organizer validator and the registered
team ID. The workspace has about 19 GiB free versus approximately
184 GB for the full EPIC validation download reported in the guide.
A larger external volume or staged download process is needed.

[Implementation audit](track4-implementation-audit.md) ·
[Submission checklist](../submission/FINAL_SUBMISSION_CHECKLIST.md) ·
[Demo script](../submission/DEMO_SCRIPT.md)

## Final-night update

The batch runner now resumes locally, preserves valid output files and
records missing inputs without writing placeholders. The current run
covers eight of 187 manifest videos. The
[submission readiness report](track4-final-submission-report.md)
supersedes the original three-video status above and gives the expanded
metrics, FPV investigation and artifact verification. The
[per-video audit](../results/final_manifest_audit.json) lists all remaining
inputs.

The [final release](dentilens-track4-final-release.md) adds the confirmed
team name, a raw-logit inspection of both FPV category misses, the public
walkthrough and the final artifact checks.
