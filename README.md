# DentiLens XR

## Team

Team name: Rajasthani

College: Vellore Institute of Technology, Vellore

| Member | Registration number |
| --- | --- |
| Arham Choudhary | 25BIT0208 |

DentiLens XR is a dental video review prototype. The intended experience is to
upload examination footage, identify visible dental regions, create timestamped
review bookmarks, and produce concise highlights from the source video.

## Working implementation

The Python pipeline decodes a CFR MP4, samples frames, runs a selected dental
region engine, and saves model observations
with confidence, bounding boxes, source frame indices and timestamps. Frames with
observations receive annotated evidence images. Observation-based highlights
are selected chronologically within 15% of the source duration and exported
as a video-only MP4.

AlphaDent's classes cover abrasion, fillings, crowns and annotated caries
categories. The tooth-region engine below adds a separate localization option. Instrument
tracking and procedure-action recognition remain development tasks. All model observations require professional
review. The prototype's clinical accuracy has yet to be evaluated.

## Set up on macOS

Use Python 3.11 in a virtual environment. The pinned dependencies are the versions
used for CPU execution on Apple Silicon.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt -r requirements-dev.txt
python -m pytest -q
```

### Local upload and review app

Install the app dependencies in the same environment as the CLI:

```bash
python -m pip install -r requirements-app.txt
bash run_app.sh
```

Open http://127.0.0.1:8501. Upload a CFR MP4, choose an installed model, and run
analysis. The review tab shows timestamped evidence, the original source for
new uploads, highlights and report downloads. All analysis runs locally. Uploaded
footage and results are stored under ignored `runtime/` and `results/` folders.
The app binds to the loopback address and is intended for local demonstration.
It does not connect to XR hardware or provide clinical diagnoses.
On this development checkout, double-click `Launch DentiLens.command` to start
the app. The text-guided engine remains the demo default. The trained detector
is explicitly experimental because the video checks exposed non-tooth predictions.

### Train a detector using the supplied intraoral dataset

The user-supplied Roboflow export is attributed to the **yusma / Intraoral Tooth
Detection**, version 2 project:
https://universe.roboflow.com/yusma/intraoral-tooth-detection-bj7bf/dataset/2.
Its included metadata declares CC BY 4.0. Source attribution and export notes
are preserved during preparation. The original images are not redistributed here.

The preparation script retains the supplied train/validation/test splits and
converts enclosing polygon coordinates to detection boxes. All source tooth
classes, including two unusual singleton labels, are combined into `tooth`.
This is a detector-training task: the export's rectangular outlines are not used
as anatomical mask ground truth. Training does not establish FDI numbering.

```bash
python scripts/prepare_intraoral.py /path/to/intraoral.yolov8.zip data/intraoral
python scripts/download_tooth_backbone.py
python scripts/train_tooth_detector.py data/intraoral/data.yaml models/yolov8n.pt runtime/tooth-training
python scripts/evaluate_tooth_detector.py runtime/tooth-training/weights/best.pt data/intraoral/data.yaml runtime/tooth-test
python scripts/install_tooth_detector.py runtime/tooth-training
python scripts/download_grounded_models.py --only sam2
bash run.sh /path/to/dental-video.mp4 results/tooth-review --engine tooth-sam2
python scripts/verify_analysis.py /path/to/dental-video.mp4 results/tooth-review
```

Training uses a pretrained YOLOv8n backbone with its first ten layers frozen,
one tooth class, 416-pixel images and validation-based checkpoint selection.
Evaluation uses the held-out test split. Hash and source-basename checks reject
exact train/test overlap, but patient-level independence cannot be established
from this export. See the [Ultralytics training documentation](https://docs.ultralytics.com/modes/train/)
and [licensing information](https://www.ultralytics.com/license).

The `tooth-sam2` engine uses each predicted tooth box to prompt SAM 2.1 tiny,
then saves masks through the same timestamped report pipeline. SAM boundaries
require independent evaluation; detection mAP is not segmentation accuracy.
Local detector weights require a matching `models/tooth-detector.json` provenance
file. Preparation, training and test results are recorded under `reference/`.

The completed run used all 2,540 training images for twelve total epochs, with
a checkpoint resume after an execution interruption. Validation selected epoch
12. The fixed checkpoint evaluated on 120 test images containing 2,011 annotated
teeth achieved **box mAP50 0.9518** and **box mAP50–95 0.5876**. Precision and
recall at Ultralytics' F1-selected operating point were 0.9427 and 0.9415.
These metrics do not measure video or segmentation accuracy. The records are
`reference/intraoral-detector-training.json`, `reference/intraoral-detector-test.json`
and `reference/intraoral-training.csv`, with the validation chart alongside them.

Five real-video runs at the fixed 0.35 threshold exposed a domain gap. The braces
clip retained only two predictions, which visual inspection identified as cheek
and glove regions. Other reviewed clips produced glove, ear and equipment
false positives. The clear examination close-up included some real tooth masks,
but missed visible teeth in other inspected frames. A separate three-frame crop experiment improved some
tooth localization while still missing teeth and including non-tooth areas. It
was not adopted as a new inference engine. No video accuracy score is claimed.
See `reference/trained-tooth-video-tests.json`. The app preserves the text-guided
baseline as its default and labels the trained engine experimental.

### Legacy AlphaDent engine

```bash
python scripts/download_model.py
bash run.sh /path/to/dental-video.mp4 results/first-analysis --engine alphadent --device cpu
```

The CLI supports `--sample-fps` and `--confidence`. Its defaults are one sampled
frame per second with the text-guided engine at confidence 0.50. AlphaDent and
the experimental tooth detector default to 0.35. A 16 GB Mac M4 is the target
development machine. CPU execution has been verified. Apple MPS availability and
model execution should be checked on the local machine before selecting `--device mps`.
All CLI runs require a new output directory. For the demo narrative and capability
boundaries, see `reference/demo-script.txt`.

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

Four additional Pexels clips were downloaded for separate smoke tests using the
same 0.50 confidence threshold, one sampled frame per second and CPU inference.
Source links, creators, input hashes and aggregate results are recorded in
`reference/pexels-tooth-region-tests.json`. Local footage and evidence images
are excluded from Git. The [Pexels license](https://www.pexels.com/license/)
allows free use and modification; the footage does not imply endorsement.

The portrait cleaning clip (33337556) exposed a segmentation failure: its retained
mask included much of the white mouth retractor instead of isolating teeth.
The wider cleaning shots (5099636 and 4490363) produced no retained observations;
their inspected middle frames showed small or obscured teeth. These results do
not establish reliable tooth segmentation. Detection counts and model confidence
are not accuracy measurements. The first three clips are also too short to fit
a four-second highlight within the 15% duration budget.

The 42.16-second close-up (6502766) yielded 16 region observations across 43
sampled frames and a four-second highlight. Inspected overlays at 1 and 10 seconds
follow exposed front-tooth groups; the overlay at 36 seconds is uncertain beside
strong mirror glare. This is a useful additional demo input, with manual review
still required. No per-clip threshold tuning was applied to these four tests.

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

## Experimental mirror review

Choose **Teeth and dental mirror head (experimental)** in the app, or pass
`--engine teeth-mirror` to the CLI. Green contours show predicted tooth regions.
Blue contours show candidate round mirror heads. The mirror model uses a
separate text prompt and a box shape filter. It can miss mirrors or outline
other objects. Tool type coverage is currently limited to round mirror heads.

`scripts/sample_video_labels.py` exports local frames and a manifest with source
frame numbers and timestamps for annotation. The export marks every frame
unlabeled. For the next training round, label visible teeth and instruments and
reserve separate source videos for evaluation.

## Experimental tooth identity and cavity review

Select **Tooth type names**, **Tooth numbers**, or **Possible cavity flags** in
the analysis picker after installing the corresponding local checkpoint. Tooth
type names appear above predicted contours. FDI mode draws labeled boxes and
shows the suggested number and tooth name together in the timeline. Use
standard-orientation footage and verify the patient's side and jaw before
saving a reviewer label. Cavity mode shows orange boxes in the **Possible
cavities** lane. Candidates must overlap a predicted tooth region, which can
suppress flags when teeth are missed. Review the original footage even when no
flag appears. Reviewer confirmations, notes and exports work with all modes.

Prepare each ZIP with `scripts/prepare_intraoral.py ARCHIVE OUTPUT` and the
appropriate option from the table. Use the prepared `OUTPUT/data.yaml` with
`scripts/train_tooth_detector.py`, then evaluate the checkpoint with
`scripts/evaluate_tooth_detector.py` and install the completed run with
`scripts/install_tooth_detector.py RUN_DIRECTORY`. Match the evaluation image
size to the training image size using `--imgsz`.

| Preparation option | Source archive and attribution | Model output |
| --- | --- | --- |
| `--tooth-types` | [Intraoral Tooth Detection v2 by yusma](https://universe.roboflow.com/yusma/intraoral-tooth-detection-bj7bf/dataset/2) | Eight tooth types |
| `--task fdi` | [Intraoral Tooth Numbering FDI v1 by DentalMate6v](https://universe.roboflow.com/dentalmate6v/intraoral-tooth-numbering-fdi/dataset/1) | 32 permanent FDI positions |
| `--task cavity` | Cavity Dataset release by CAV-TEE Research Team | Candidate cavity boxes |

Each export declares CC BY 4.0. The preparation script converts the source
annotations to enclosing boxes and retains the original splits. FDI excludes
codes `327` and `56` from the 32-class task. Cavity preparation normalizes
pixel coordinates and records invalid boxes that were omitted. The two
intraoral exports share source image names and split assignments, so they
should not be treated as independent validation datasets. The cavity test
split includes only five images with cavity annotations. Keep the existing
text-guided mode as the demo default until improved video performance can be
demonstrated with annotated footage.
