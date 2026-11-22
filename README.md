# EgoSight XR

**See the Actions. Understand the Moment.**

EgoSight XR lets you review first-person recordings by action and jump straight to a selected highlight. Team Rajasthani built it for Samsung R&D Institute India XR Hackathon 2026, Track 4.

## Samsung Track 4 submission

[Submission files](submission/EgoSight%20XR) · [Model and CLI setup](track4) · [Verification results](reference/egosight-verification.md)

The staging ZIP preserves the organizer’s order: `run.sh`, `code/`, `weights/`, `requirements.txt`, `outputs/`, `README.md`. The registered TEAM_ID and checkpoint redistribution terms still need confirmation. The ZIP uses the development ID and includes checkpoint setup instructions.

## Start the workspace

On the development Mac, run `bash run_full.sh` and open [EgoSight XR](http://127.0.0.1:8502/). For a fresh checkout, use Python 3.11 and Node.js, install the frontend and Python dependencies, then set up the [Track 4 weights](track4/weights/README.md).

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-full.txt
npm --prefix frontend ci
npm --prefix frontend run build
bash run_full.sh
```

- **Video Intelligence** accepts MP4s up to 500 MB and at least 4 seconds long. Preview the upload, run inference, seek from the action list or timeline, and play the selected source intervals.
- **Analysis History** reopens locally stored recordings and validated results.
- **Benchmark & Validation** shows the Samsung manifest, canonical labels and the team’s local checks.
- **Settings** shows whether the local server and checkpoint files are available.

## How inference works

The React workspace calls the local FastAPI server. OpenCV reads the upload and the server invokes the standalone Python Track 4 runner. Known EPIC-KITCHENS clips use a pretrained TSN RGB ResNet50 for timed verb–noun predictions. Their activity category comes from the benchmark contract. Known FPVSum clips and other MP4 uploads use a pretrained R3D-18 category-transfer model. This path returns an activity category and highlights, and may return `other` where evidence is weak. The two paths share highlight selection and local JSON checking.

Completed runs are stored under the app’s local runtime. The browser displays the source video and the genuine model output, and its download serves the exact JSON file from the runner. For an independent run, see [Track 4 inference and validation](track4). Browser uploads outside the manifest are development runs.

Team Rajasthani · Vellore Institute of Technology, Vellore
