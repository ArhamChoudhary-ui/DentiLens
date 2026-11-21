# DentiLens XR — Action Intelligence Engine

Samsung XR Hackathon Track 4 implementation inside the DentiLens repository.
Team Rajasthani · Vellore Institute of Technology, Vellore.

First-person recordings can make brief actions difficult to find. DentiLens
reuses its video processing and review infrastructure to predict timed actions
and export selected original footage. The action benchmark uses its own
pretrained models. OpenCV decodes the video, PyTorch scores frames, the
selector ranks temporal segments, and FFmpeg exports the chosen clips.

The local run processed **8 of 187 manifest videos** and validated every one.
The remaining **179 source videos** were unavailable in the supplied
workspace. The organizer's validator and official scores remain unavailable.

## Results and presentation

- [Public walkthrough MP4](https://raw.githubusercontent.com/ArhamChoudhary-ui/DentiLens/xr-4-video-intelligence/reference/submission/DentiLens-Track4-Walkthrough.mp4)
- [PowerPoint](https://github.com/ArhamChoudhary-ui/DentiLens/blob/xr-4-video-intelligence/reference/submission/DentiLens-Track4-Presentation.pptx) and [PDF](https://github.com/ArhamChoudhary-ui/DentiLens/blob/xr-4-video-intelligence/reference/submission/DentiLens-Track4-Presentation.pdf)
- [Final release report](https://github.com/ArhamChoudhary-ui/DentiLens/blob/xr-4-video-intelligence/reference/dentilens-track4-final-release.md)
- [Actual EPIC and FPV JSON samples](reference/samples)

Official weights are action coverage 40%, action mAP 25%, highlight F1
25% and video-level label accuracy 10%. Local diagnostics use public EPIC
action annotations and FPVSum frame scores after inference. They report
temporal overlap, a distinct-pair coverage proxy and frame-level F1 on
the available subset. They do not establish an official score.

## Run inference

Set up Python 3.11 in this directory and install `requirements.txt`.
Place the two named checkpoints in `weights/` before running inference.
Verify the documented upstream sources and the SHA-256 digests in
`weights/receipt.json`. The CLI makes no implicit network requests.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export DENTILENS_TEAM_ID='your-registered-team-id'
./run.sh "/path/to/P03_26.mp4" "/path/to/outputs/P03_26.json"
.venv/bin/python scripts/check_submission.py --file "/path/to/outputs/P03_26.json"
```

The two-argument command reads the registered ID from the environment or
`team_id.txt`. The output name must match the input stem. The default run
requires exact manifest membership and duration within 0.5 seconds.
On a different development clip, supply `--format review --dataset
epic_kitchens_100` or `--format review --dataset fpvsum`.

Use `--report /path/to/evidence.json` to record per-window alternatives,
checkpoints, timing and highlight selection. Use `--highlight-mp4
/path/to/highlight.mp4` for a video-only re-encode of the original selected
segments. Keep review evidence and highlight MP4 files out of `outputs/`.
Use `--device cpu` for the tested Mac path. The generic device selector
also supports explicitly available CUDA or MPS.

## Models and limitations

- EPIC-KITCHENS-100 TSN RGB ResNet50 scores 97 verbs and 300 nouns on
  sampled RGB frames. One decoder pass caches 2 Hz logits. Two- and
  four-second windows aggregate logits and choose among pairs present in
  **training** annotations. Pair-wise temporal NMS removes overlapping
  duplicates. These are window estimates and there is no boundary head.
- FPVSum uses a Kinetics-400 R3D-18 category-transfer baseline with 11
  mapped categories and an explicit low-evidence fallback. Bikepolo,
  longboarding and plane lack a justified mapping. The FPVSum path emits
  empty verb–noun actions, which the guide allows.
- Highlight selection greedily favors semantic novelty and action coverage,
  with confidence, motion and a redundancy penalty. It uses original
  non-overlapping chronological cuts under the 15% target when permitted
  by the four-second minimum. For three short manifest videos the rules
  conflict. The implementation emits one four-second clip with a warning.
- A trained VideoMAE-head route is retained behind `--checkpoint`. Motion
  and infrastructure modes remain explicitly marked development baselines.
- The implementation has no independently verified hand-object contact
  reasoning, no official combined benchmark score and no headset deployment.

## Data, output and permissions

Get authorized source clips from the [official EPIC-KITCHENS downloader](https://github.com/epic-kitchens/epic-kitchens-download-scripts)
or the source locations indexed by [FPVSum](https://github.com/azuxmioy/fpvsum).
Check the [Bristol footage terms](https://data.bristol.ac.uk/data/dataset/2g1n6qdydwa9u22shpxqzp0t8m)
and the terms for individual FPV videos. The public release exports
predictions and a timeline walkthrough. The original recordings and model
checkpoints are acquired separately. See [checkpoint setup](weights/README.md).

The written organizer schema uses six fields, `team_id`, `video_id`,
`duration_sec`, `label`, `actions` and `highlights`. Action records
contain `segment`, `verb`, `noun` and `score`. Highlight records contain
only `segment`. The separate review format retains highlight importance.
Use `DEVELOPMENT` only to reproduce the public samples.

## Organizer kit and batch validation

The supplied manifest and vocabulary are unchanged. The manifest has 138
EPIC-KITCHENS and 49 FPVSum targets. Both kit copies omit organizer
`validate.py`. `scripts/check_submission.py` is a separate local checker.
The illustrative `skiing` label conflicts with the supported `ski` label.
Runtime category mapping uses `ski` and preserves the actual video ID.

```bash
.venv/bin/python scripts/run_manifest.py /path/to/videos /path/to/outputs --team-id "$DENTILENS_TEAM_ID" \
  --report /path/to/audit.json --check-only
.venv/bin/python scripts/run_manifest.py /path/to/videos /path/to/outputs --team-id "$DENTILENS_TEAM_ID" \
  --report /path/to/audit.json
.venv/bin/python scripts/check_submission.py --dir /path/to/outputs
.venv/bin/python scripts/package.py /path/to/outputs /path/to/final.zip --team-id "$DENTILENS_TEAM_ID" \
  --organizer-validator /path/to/official/validate.py
```

The batch runner searches recursively for manifest-named MP4s and uses one
CPU inference process at a time. It reuses valid JSON, keeps invalid output
files for inspection, and continues after individual video failures. Missing
sources and stale or duplicate output files appear in the separate audit.
Re-run the same command to resume after adding more media. A nonzero exit
status signals an incomplete or inconsistent set. Use a fresh output folder
when the registered team ID replaces the development ID.

Final packaging requires every output and runs the supplied validator.
`--development` produces a separately marked local archive for smoke
testing. Add `--staging-name Rajasthani --external-checkpoints` to build
the shareable staging copy with source, hashes and valid development
outputs. Acquire both pretrained checkpoints before offline inference.
