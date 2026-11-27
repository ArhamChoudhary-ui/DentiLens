# EgoSight XR — Action Intelligence Engine

Samsung XR Hackathon Track 4 implementation for EgoSight XR.
Team Rajasthani · Vellore Institute of Technology, Vellore.

Open **Video Intelligence** in EgoSight XR to upload and review an MP4. The browser uses the same inference engine as the standalone CLI. OpenCV decodes video, PyTorch scores RGB windows, the selector ranks temporal segments, and FFmpeg can export selected footage through the CLI.

The [submission folder](../submission/EgoSight%20XR) contains the current staging snapshot. The organizer’s validator and official scores are unavailable.

## Demonstration and examples

- [Submission materials](../submission/EgoSight%20XR)
- [Verification report](../reference/egosight-verification.md)
- [Real JSON samples](reference/samples)

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
export EGOSIGHT_TEAM_ID='your-registered-team-id'
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
.venv/bin/python scripts/run_manifest.py /path/to/videos /path/to/outputs --team-id "$EGOSIGHT_TEAM_ID" \
  --report /path/to/audit.json --check-only
.venv/bin/python scripts/run_manifest.py /path/to/videos /path/to/outputs --team-id "$EGOSIGHT_TEAM_ID" \
  --report /path/to/audit.json
.venv/bin/python scripts/check_submission.py --dir /path/to/outputs
.venv/bin/python scripts/package.py /path/to/outputs /path/to/final.zip --team-id "$EGOSIGHT_TEAM_ID"
```

The batch runner searches recursively for manifest-named MP4s and uses one
CPU inference process at a time. It reuses valid JSON, keeps invalid output
files for inspection, and continues after individual video failures. Missing
sources and stale or duplicate output files appear in the separate audit.
Re-run the same command to resume after adding more media. A nonzero exit
status signals an incomplete or inconsistent set. Use a fresh output folder
when the registered team ID replaces the development ID.

Final packaging requires all 185 scored manifest outputs and a verified team ID.
Samsung confirmed that this local checker can be used for pre-submission
verification. The two excluded FPVSum IDs stay in the supplied manifest.
Pass an organizer validator only if one was supplied.
`--development` produces a separately marked local archive for smoke
testing. Add `--staging-name Rajasthani --external-checkpoints` to build
the shareable staging copy with source, hashes and valid development
outputs. Acquire both pretrained checkpoints before offline inference.

### Acquire and process a partial batch

`scripts/acquire_manifest.py` uses the organizer manifest and the upstream
EPIC and FPVSum source lists. It reuses verified JSON and partial EPIC
downloads in the same directories. The six-worker configuration below is
optional. Video inference runs one file at a time. Successful managed
downloads are removed after their outputs pass validation.

To run it, collect `epic_55_splits.csv`, `md5.csv`, and `errata.csv`
from the [EPIC-KITCHENS downloader](https://github.com/epic-kitchens/epic-kitchens-download-scripts/tree/master/data)
and save [FPVSum's video list](https://github.com/azuxmioy/fpvsum/blob/master/crawler/FPVSum_videolist.csv)
as `fpvsum-videolist.csv` in the same metadata directory. Install `yt-dlp`
if the selected work includes FPVSum video downloads.

```bash
.venv/bin/python scripts/acquire_manifest.py /path/to/student-kit /path/to/metadata \
  /path/to/outputs /path/to/downloads --team-id DEVELOPMENT --target-count 100 --epic-workers 6
```

Monitor the local `acquisition.json` in the download directory. The
target counts all valid outputs already in the output directory. A
100-video run is partial coverage of the 187-video organizer manifest.

## Optional TSN calibration experiment

The trainer freezes the existing TSN ResNet-50 and learns two temperature
parameters and 397 class offsets. It verifies the official training annotation
file and source-video checksums, separates participants across splits, and logs
held-out results against the unchanged logits on the same windows.

Run from the repository root with the six complete official videos named
`<video_id>.mp4` and the original `EPIC_100_train.csv` and downloader
`md5.csv`. The experiment uses official EPIC training footage and excludes
Samsung manifest IDs.

```bash
.venv/bin/python track4/scripts/train_tsn_calibration.py \
  /path/to/EPIC_100_train.csv /path/to/training-videos runtime/tsn-calibration.pt \
  --checksums /path/to/md5.csv \
  --train P02_116 P08_07 P06_109 P26_15 --val P03_116 P04_103
```

Use `--tsn-calibration /path/to/checkpoint.pt` to opt in explicitly on EPIC
inference. The normal browser and two-argument submission command use the
original TSN weights and need no calibration checkpoint. Held-out windows are
independent for the calibration layer. The upstream TSN may already have seen
them during pretraining. This comparison is a development check for verb and
noun classification on annotated windows. It does not measure localization,
FPVSum activity recognition or official Track 4 scores.

[Recorded calibration results](../reference/tsn-calibration-results.json) from the six-source CPU experiment.
