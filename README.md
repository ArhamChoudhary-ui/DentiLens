# DentiLens XR

## Samsung XR Hackathon · Track 4

Team **Rajasthani**, Vellore Institute of Technology, Vellore, built the
DentiLens XR Action Intelligence Engine for first-person video summarization.
Samsung’s required submission archive layout is shown below in order.

```text
TEAM_ID/
├── run.sh
├── code/
├── weights/
├── requirements.txt
├── outputs/
│   └── <video_id>.json
└── README.md
```

`TEAM_ID` is the identifier issued by Samsung. The staging package uses
`Rajasthani_Staging` while that identifier is unverified.

[Track 4 code and setup](https://github.com/ArhamChoudhary-ui/DentiLens/tree/xr-4-video-intelligence/track4) ·
[Final presentation](https://github.com/ArhamChoudhary-ui/DentiLens/tree/xr-4-video-intelligence/reference/submission/Final%20Presentation) ·
[Package audit](https://github.com/ArhamChoudhary-ui/DentiLens/blob/xr-4-video-intelligence/submission/PACKAGE_AUDIT.md) ·
[Submission checklist](https://github.com/ArhamChoudhary-ui/DentiLens/blob/xr-4-video-intelligence/submission/FINAL_SUBMISSION_CHECKLIST.md)

## Team

Team name: Rajasthani

College: Vellore Institute of Technology, Vellore

Members are Arham Choudhary and Aadeesh Sanghi.

DentiLens XR turns a dental examination video into a full-length annotated
recording with searchable tooth tracks, evidence frames and reviewer-controlled
bookmarks.

## Full-video examination workspace

On the development Mac, double-click `Launch DentiLens XR.command` in Finder or
run the following from the repository. Leave Terminal open during the demo.

```bash
bash run_full.sh
```

Open [DentiLens XR](http://127.0.0.1:8502/). Start a new examination, choose an
MP4, MOV, M4V or WebM file, and pick Fast, Balanced or Detailed analysis.
Use the Annotated and Original controls to compare the same timestamp. Search
for a tooth type or FDI number, select a timeline suggestion, and confirm,
correct or dismiss it. You can also add a manual bookmark at the current
playback position. Patient view shows the confirmed moments and reviewer notes.
Download the full annotated MP4, evidence images, reviewed reel, JSON report
or frame annotations from the workspace. The reviewed reel briefly holds
the annotated bookmark frames and is silent.

XR preview simulates the display on the uploaded recording. **Bookmark this**
saves the active timestamp. Try the command field or the microphone when your
browser supports on-device speech recognition in English. The prototype uses
prerecorded footage and saves the same bookmarks as the review workspace.

### Review corrections and export

- Open **FDI number and region** to confirm a permanent FDI number or draw a
  box on the original frame. The report records both the previous AI
  suggestion and the review history.
- Open **Tooth chart** to browse permanent teeth in patient orientation.
  A confirmed bookmark appears as a confirmed tooth only if it has a saved
  FDI number.
- In **Export full video**, choose suggestions or confirmed bookmarks.
  Standard, minimal and detailed overlays all preserve the playback timeline
  and audio stream. Corrections follow a linked individual tooth track. A
  manually drawn box applies at its bookmark frame.
- In **Patient view**, select a confirmed bookmark and choose a second approved
  moment to compare. XR preview supports typed navigation, speech where the
  browser supports local recognition, and a note for the next bookmark.
- Use **Stop analysis** to stop at the next progress checkpoint. Use
  **Delete examination** to remove its local footage, notes and outputs.

The [enhancement audit](reference/enhancement-audit.md) records the original
pipeline. The [video measurements](reference/enhancement-validation.json)
cover five local clips. Frame reasons describe what the pipeline observed.
Independent labels are needed to measure eligible-frame coverage.

To measure eligibility from independently annotated footage, create a JSON
array with one object per normalized frame in order. A record can look like
`{"frame":0,"visibility":"partially_visible","eligible":true}`. Supported
visibility values are `clear`, `partially_visible`, `absent`, `occluded`,
`blurred`, `poor_lighting`, `camera_motion`, and `uncertain`.

```bash
python scripts/evaluate_frame_coverage.py /path/to/timeline.jsonl /path/to/visibility.json
```

### Explore the 3.0 evidence index

- In **Tooth chart**, select a number to see the matching numbered frames,
  appearance ranges and review records. Open **Frame observations** to
  inspect scores or load the full local track.
- In **Search & quality**, search for a tooth name, FDI number, timestamp,
  notes, or the clearest saved evidence. Each result opens its source time.
- Select **Scan recording** for image-quality cues derived from the entire
  recording. Dark, bright, low-detail and large-change cues are image
  measurements. Assess individual tooth visibility in the original video.
- Download the **Evidence index** to inspect explicit frame, track, evidence
  and review records. The index is built from existing session files
  and live reviews and uses the same SQLite schema.

See the [3.0 repository audit](reference/xr3-audit.md) for the Track 4
package findings and model dependencies.

### Use the 3.5 instrument and anatomy features

- Open **Instruments** and choose **Mark instrument** to draw a box manually.
  Choose **Show model candidates** after scanning to inspect the experimental
  results. Enter your own label before confirming. Redraw a candidate box if the
  region is wrong, or dismiss it.
- **Search & quality** and **Include reviewed instruments** in the full-video
  export use confirmed regions. The evidence index retains the original model
  candidates for audit.
- Open **3D anatomy** or select a position in **Tooth chart** and choose
  **View anatomy**. Drag to rotate, zoom, and use the structure selector
  to isolate a tooth, gum or jaw. The model is educational. It provides
  28 permanent teeth and source attribution.

Install the pinned [OrganUI anatomy model](https://github.com/organui/teeth-3d)
using the command below before building. The GLB and its attribution retain
the BodyParts3D / Anatomography CC BY-SA 2.1 Japan license. Files are served
locally from the app after installation.

```bash
python3 scripts/install_anatomy.py
npm --prefix frontend ci
npm --prefix frontend run build
bash run_full.sh
```

The instrument model uses the user-supplied [Dental Instruments v30
export](https://universe.roboflow.com/object-detect-k48yz/dental-instruments)
and a local YOLOv8n OBB backbone. Raw images and trained checkpoints stay
local. Reproduce training with these commands after installing the full
backend dependencies and obtaining the official `yolov8n-obb.pt` backbone.

```bash
python scripts/prepare_instruments.py "/path/to/Dental Instruments.v30i.yolov8-obb.zip" data/instrument-v30
python scripts/train_instruments.py data/instrument-v30 /path/to/yolov8n-obb.pt runtime/instrument-fit --epochs 3
python scripts/train_instruments.py data/instrument-v30 runtime/instrument-fit/fit/weights/best.pt runtime/instrument-refine --epochs 8 --install models
```

See the [dataset registry](datasets/dataset_registry.yaml),
[technical audit](reference/xr35-validation.md) and
[held-out photo evaluation](reference/instrument-model-v30-evaluation.json).

### Review instruments, audio and chapters in XR 4.0

- Confirm a box in **Instruments**, then choose **Follow nearby frames**.
  Inspect each proposed box against the source. Choose **Review this frame**
  to place a confirmed observation and preserve its link to the anchor.
  Download review records and original candidate boxes as JSON.
- In **Audio**, select a language and transcribe the source recording locally.
  Jump to a segment, edit text or timing, add a note or associate an FDI number.
  You can also record a voice note, edit the transcription, and save it as a
  bookmark at the selected source time.
- In **Storyboard**, build chapters from source frames and timestamped
  evidence. Seek by thumbnail, review or edit a chapter, and export CSV or JSON.
  Use **Search & quality** to find transcript text and chapter titles.

Install the pinned [Whisper tiny model](https://huggingface.co/openai/whisper-tiny)
for local speech recognition with `python scripts/install_whisper.py`.
The download requires an internet connection. Inference runs on CPU with
local files. Videos without audio are reported as such.

See the [video evaluation](reference/xr40-audit.md) for source-level splits,
measured errors, and the reason the video-trained model remains experimental.

### Install on a fresh checkout

Use Python 3.11 and Node.js 22. The app prefers PyTorch MPS when available and
uses CPU as a fallback. FFmpeg is supplied by the pinned Python dependency.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-full.txt
python scripts/download_grounded_models.py --only dino
npm --prefix frontend ci
bash run_full.sh
```

Install the locally trained `tooth-fdi`, `tooth-types` and `cavity-review`
checkpoint/provenance pairs in `models/` as described in
[model setup](models/README.md) and
[the training section](#experimental-tooth-identity-and-cavity-review).
The repository contains training and install scripts. Model weights, uploaded
videos and the examination database are stored locally. SQLite sessions and
exports live under `runtime/full-video/`.

The worker converts the full recording to browser-ready H.264 with source
audio when available. It limits the long edge to 1280 pixels and sampling to
30 fps. Fast, Balanced and Detailed run the models approximately every 2,
1 and 0.5 seconds respectively, with early refresh on track loss or scene
cuts. ByteTrack associates boxes, checked optical flow bridges between
detections, and repeated identities are needed before an FDI number appears.
The full-length export preserves sections where a track has no reliable
geometry. Source metadata and actual overlay coverage are reported separately.

The current models predict boxes on optical dental footage. FDI numbers and
tooth types are suggestions for reviewer verification. Cavity flags are
optional research outputs. Recordings with zero flags still require ordinary review. Review decisions attach to bookmarks, while the full annotated
recording preserves the original model suggestions.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
npm --prefix frontend run build
```

## Legacy snapshot workflow

The original Python pipeline decodes a CFR MP4, samples frames, runs a selected dental
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

### Legacy Streamlit upload and review app

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

The Action Intelligence Engine runs offline through
[`track4/run.sh`](track4/run.sh). It produces timed verb–noun actions and
selects highlights from the original first-person video. The
[Track 4 README](track4/README.md) describes the separate setup,
manifest runner and validation commands.

See the [final release report](reference/dentilens-track4-final-release.md),
[manifest audit](results/final_manifest_audit.json) and
[submission checklist](submission/FINAL_SUBMISSION_CHECKLIST.md),
[package audit](submission/PACKAGE_AUDIT.md) and
[AI disclosure](AI_DISCLOSURE.md).
Team Rajasthani verified eight outputs locally. See the
[public walkthrough](reference/submission/DentiLens-Track4-Walkthrough.mp4),
[PowerPoint](reference/submission/DentiLens-Track4-Presentation.pptx) and
[PDF](reference/submission/DentiLens-Track4-Presentation.pdf).
The organizer's [presentation template](reference/submission/Submission_Final_forGithub.pptx)
is preserved with the original layout.

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
shows the suggested number and tooth name together in the timeline. Grounding
DINO crops around the highest scoring tooth region before FDI prediction. Use
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

### Recorded demo checks

The measured photo results and video smoke tests are in
[the evaluation record](reference/new-dental-detector-results.json). The
eight-type model reached 75.6% box mAP50 on 120 photos. The 32-class FDI
model reached 65.8% box mAP50 on 120 photos. Cavity-class AP50 was 27.6%
on the cavity dataset's 38-image test split. These are photo detection
metrics for the underlying models. The app applies additional processing
to video and has no measured video accuracy.

In this local checkout, open **tooth-fdi-braces-demo** at 20 seconds for
FDI 11 and 21 suggestions, or **tooth-types-review-demo** at 14 seconds for
incisor suggestions. The cavity model initially flagged gloves and a
mirror on the exam clip. The tooth overlap filter removed all five
candidates in the rerun. The annotated photo spot check still showed both
extra flags and missed labeled areas.

### Export a playable annotated review

Under **Review saved results**, select a moment and choose **Build annotated
reel**. The MP4 plays source footage and briefly holds each annotated source
frame. Its label panel lists the model suggestions and any saved reviewer
decisions, notes and flags. Overlays are drawn only on their observed frames,
so they do not follow moving teeth between samples. Choose **Confirmed
moments** to include saved manual bookmarks or **AI highlights** for the
suggested windows. Dismissed predictions are excluded. The original highlight
remains available under **Original AI highlight**. Exports are silent.

If the number list is crowded, choose a single prediction in the timeline.
The close-up displays its selected box and full suggested tooth name. Source
footage may be unavailable for legacy results without a linked MP4. Their
saved evidence and notes remain accessible. When no predictions pass the
threshold, you can still save a timestamp as a manual bookmark and include
it in a confirmed reel.

If the browser reports a refused connection, start `bash run_app.sh` in
Terminal from the repository directory and keep Terminal open. Then load
`http://127.0.0.1:8501/`. Double-clicking the launcher in Finder also runs
it. Opening the launcher as a file preview displays its script.

### Names, numbering, and possible cavities in one run

Choose **Tooth names, FDI numbers and possible cavities** after uploading an
MP4, or pass `--engine combined-review` to the CLI. The mode requires the three
local detector checkpoints and Grounding DINO. One tooth-region pass supplies
the crop for the two identity models and filters cavity candidates.

When the models cover the same tooth and agree on its type, the review shows the
suggested FDI number and anatomical name together. The review withholds the
number when the two types disagree. A type-only observation still appears if
the numbering model misses it. Orange boxes are possible cavity flags. An
overlapping tooth suggestion appears only when the association is unambiguous
by box overlap. Verify both suggestions during review.

Use **All findings**, **Teeth**, or **Possible cavities** in the timeline. Pick
**All findings at selected frame** in the reel builder to export tooth and cavity
suggestions together when they appear on that frame.

The local **combined-braces-demo** shows FDI 11 and 21 suggestions at 20 seconds.
On that clip the combined mode produced six tooth suggestions across 14 sampled
frames, versus three for FDI alone. No cavity flags passed the filter. The
[smoke test record](reference/combined-review-smoke-test.json) also reports a
five-photo geometry check. The combined mode reuses the existing checkpoints.
These checks do not establish clinical accuracy or improved video accuracy.
