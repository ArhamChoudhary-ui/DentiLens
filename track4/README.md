# DentiLens XR — Action Intelligence Engine

Samsung XR Hackathon Track 4 implementation inside the DentiLens repository.
The root dental app and its entry points continue to operate independently.

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
python scripts/check_submission.py --file "/path/to/outputs/P03_26.json"
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

## Organizer kit and batch validation

The supplied manifest and vocabulary are unchanged. The manifest has 138
EPIC-KITCHENS and 49 FPVSum targets. Both kit copies omit organizer
`validate.py`. `scripts/check_submission.py` is a separate local checker.
The illustrative `skiing` label conflicts with the supported `ski` label.
Runtime category mapping uses `ski` and preserves the actual video ID.

```bash
python scripts/run_manifest.py /path/to/videos /path/to/outputs --team-id "$DENTILENS_TEAM_ID" --check-only
python scripts/run_manifest.py /path/to/videos /path/to/outputs --team-id "$DENTILENS_TEAM_ID"
python scripts/check_submission.py --dir /path/to/outputs
python scripts/package.py /path/to/outputs /path/to/final.zip --team-id "$DENTILENS_TEAM_ID" \
  --organizer-validator /path/to/official/validate.py
```

Final packaging requires every output and runs the supplied validator.
`--development` produces a separately marked local archive for smoke
testing. Keep original video and restricted model assets out of public Git.
