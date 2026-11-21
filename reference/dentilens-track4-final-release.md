# DentiLens XR Track 4 release

**Team Rajasthani · Vellore Institute of Technology, Vellore**

DentiLens XR turns first-person MP4s into searchable action intervals and
short highlights. TSN RGB scores kitchen verb–noun pairs. R3D-18 transfers
Kinetics-400 predictions to supported FPVSum categories. Selection rewards
action coverage, confidence and visual change before FFmpeg cuts the
source intervals. The existing dental app is separate from the benchmark.

## Results

- Eight of 187 organizer videos processed and locally validated. Five EPIC
  videos yielded 157 scored action intervals. Three FPVSum videos yielded
  category labels and highlights.
- On the local five-video EPIC subset, action mAP was 0.497 at tIoU 0.1 and
  0.464 at tIoU 0.5. The separate local coverage proxy captured two of
  seven per-video distinct reference pairs.
- On the local three-video FPVSum subset, category accuracy was 1 of 3
  and mean frame-level highlight F1 was 0.127. Official scores are pending.
- Three short clips kept the four-second minimum and received the
  documented budget warning. The local audit lists each interval length.

## FPVSum investigation

Both misses use supported categories. Skate maps to Kinetics
`skateboarding`. Surfing maps to `surfing water`, `windsurfing` and
`kitesurfing`. R3D-18 checks its pretrained weight hash, samples eight
RGB frames per four-second window, applies the Torchvision Kinetics
V1 transform and averages the mapped softmax scores over the video.

| Reference label | Windows | Pooled target score | Strongest pooled class | Peak target window | Output |
| --- | ---: | ---: | --- | ---: | --- |
| skate | 10 | 0.00335 | rockclimbing, 0.01614 | 0.00869 | other |
| surfing | 18 | 0.05826 | surfing, 0.05826 | 0.26997 | other |

The skateboarding logit ranged from 0.918 to 4.290 and never ranked first.
The surfing-water logit ranged from 2.191 to 16.447 and never ranked
first. High-scoring alternatives included throwing actions for skate
and sailing or jetskiing for surfing. Surfing had useful evidence in some
windows, diluted by averaging. Both pooled winners fell below the 0.10
cutoff. This points to category transfer and pooling limitations. No
mapping or preprocessing defect was established. The original
predictions reproduced byte for byte. Ground truth was used only to
evaluate them.

## Deliverables

| Item | Status |
| --- | --- |
| Code and reproduction | [Track 4 README](../track4/README.md) and [GitHub commits](https://github.com/ArhamChoudhary-ui/DentiLens/commits/xr-4-video-intelligence/) |
| Manifest and validation | [Audit JSON](../results/final_manifest_audit.json) and [per-video validation](dentilens-final-output-validation.md) |
| Release audit | [Repository and artifact checks](dentilens-final-release-audit.md) |
| Presentation | [PPTX](submission/DentiLens-Track4-Presentation.pptx) and [PDF](submission/DentiLens-Track4-Presentation.pdf) |
| Demo | [48-second public MP4](https://raw.githubusercontent.com/ArhamChoudhary-ui/DentiLens/xr-4-video-intelligence/reference/submission/DentiLens-Track4-Walkthrough.mp4). The 59-second footage version remains local. |
| Archive | Verified staging ZIP with source, reproducibility metadata and eight valid outputs. Upstream checkpoints were supplied separately in the extraction test. |
| Tests | 99 passed. Extracted archive reproduced P02_13 byte for byte. |
| Official validator and team ID | Organizer validator unavailable. Rajasthani confirmed. A separate issued TEAM_ID is unverified. |
| Google Form | Pending human submission. [Checklist](../submission/FINAL_SUBMISSION_CHECKLIST.md) and [email draft](../submission/ORGANIZER_EMAIL_DRAFT.md). |
