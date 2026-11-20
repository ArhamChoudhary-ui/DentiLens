# DentiLens XR

## Team

Team name: Rajasthani

College: Vellore Institute of Technology, Vellore

| Member | Registration number |
| --- | --- |
| Arham Choudhary | 25BIT0208 |
| Aadeesh Sanghi | 25BDS0044 |

DentiLens XR is a dental video review prototype. The intended experience is to
upload examination footage, identify visible dental regions, create timestamped
review bookmarks, and produce concise highlights from the source video.

## Working implementation

The current Python pipeline decodes a CFR MP4, samples frames, runs a published
AlphaDent dental-photo segmentation checkpoint, and saves model observations
with confidence, bounding boxes, source frame indices and timestamps. Frames with
observations receive annotated evidence images. Observation-based highlights
are selected chronologically within 15% of the source duration and exported
as a video-only MP4.

AlphaDent's classes cover abrasion, fillings, crowns and annotated caries
categories. The tooth-region engine below adds a separate localization option. Instrument
tracking and procedure-action recognition remain development tasks. All model observations require professional
review. The prototype's clinical accuracy has yet to be evaluated.

## Set up on macOS

Use Python 3.11 in a virtual environment. Dependencies are pinned to the versions
used for the recorded CPU test on Apple Silicon.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/download_model.py
python -m pytest -q
```

Run an analysis with a new output directory.

```bash
bash run.sh /path/to/dental-video.mp4 results/first-analysis --device cpu
```

The CLI supports `--sample-fps` and `--confidence`. Its defaults are one sampled
frame per second and a confidence threshold of 0.35. A 16 GB Mac M4 is the target
development machine. CPU execution has been verified. Apple MPS availability and
model execution should be checked on the local machine before selecting `--device mps`.

## Read the results

Each completed analysis creates `report.json` and an `evidence/` folder. When
observations support selected highlights, it also creates `highlights.mp4`.
The report records repeated observations per frame, model provenance and
capability limitations. Source video timing is derived from CFR frame indices.

The first supplied clip was 42.16 seconds at 4096×2160 and 25 fps. All 1054 frames
decoded, and inference sampled 43 frames. AlphaDent returned zero observations
at confidence 0.35, so the selected highlight list was empty. This establishes
pipeline execution on that clip. It provides no measured dental accuracy.

## Tooth-region segmentation

The text-grounded engine combines Grounding DINO tiny with SAM 2.1 tiny.
Grounding DINO locates visible teeth from the text prompt `teeth.`. SAM 2 predicts
masks inside those model-generated boxes. The pipeline independently processes
sampled frames and saves binary masks, overlays and normalized source coordinates.
The masks may cover several adjacent teeth. Individual numbering, temporal
tracking and procedure-action classification remain development tasks.

```bash
python -m pip install -r requirements-grounded.txt
python scripts/download_grounded_models.py
bash run.sh /path/to/video.mp4 results/tooth-review \
  --engine grounded-sam2 --confidence 0.50 --device cpu
```

The grounded engine defaults to confidence 0.50. Qualitative inspection of the
supplied braces clip at 0.35 showed lip regions being mislabeled as teeth. The
higher threshold removes those observed lower-confidence detections, while also
missing some visible-tooth frames. It is a prototype setting selected on this
clip and requires validation on separate annotated videos.

Both model revisions and weight hashes are pinned in
`reference/grounded-models.json`. The downloader verifies the weights, and model
loading rejects missing or mismatched parameters. The SAM 2.1 video checkpoint's
image-segmentation components are used here. Mask quality estimates in the report
are model outputs and require evaluation against annotated reference masks.

The downloaded optical research checkpoint produced tooth-edge predictions on
the braces footage, but its contour stage failed with empty initial regions at
both tested input resolutions. This candidate is excluded from the active engine.
Instrument prompts were also excluded after qualitative inspection showed glove
regions being included in instrument detections.

Model references are https://huggingface.co/IDEA-Research/grounding-dino-tiny and
https://huggingface.co/facebook/sam2.1-hiera-tiny.

## Recorded tooth-region test

The supplied 6528841 clip is 27 seconds at 2732×1440 and 25 fps. The pipeline
verified all 675 source frames and sampled 27 frames. At confidence 0.50, it
retained seven teeth-region observations and exported a four-second original-
footage highlight, within the 4.05-second budget. Masks and source timestamps
were checked, and selected overlays were inspected visually. The threshold
was selected using this clip. Ground-truth annotations and evaluation on separate
videos are still needed to measure accuracy. The aggregate verification record
is saved in `reference/tooth-region-smoke-test.json`.

## Samsung Track 4

The separate student-guide contract requires six-field JSON outputs for 187
manifest videos. This dental review CLI currently writes a research review
report into an analysis directory. Track 4 benchmark inference, organizer
validation and complete submission outputs remain separate pending work.
The supplied guide, manifest, vocabulary and two examples are preserved under
`reference/student_package/`. The received package lacks `validate.py`.

The supplied nine-slide presentation template is preserved unchanged at
`reference/submission/Submission_Final_forGithub.pptx`. Its cover has placeholders
for the track, team, college and member details. The content sections cover the
problem, solution, architecture, track output, differentiation and demo/setup,
followed by a "Brownie Slides" section and closing slide. The template is blank.

## Model attribution

AlphaDent by Sosnin et al. Published model release and source are available at
https://github.com/ZFTurbo/AlphaDent/releases/tag/v1.0. Checkpoint provenance and
the publisher's accompanying license are preserved under `reference/`.
