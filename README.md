# EgoSight XR

**See the Actions. Understand the Moment.**

EgoSight XR is a unified egocentric video intelligence platform developed by **Team Rajasthani** (Vellore Institute of Technology) for the **Samsung R&D Institute India XR Hackathon 2026, Track 4 (Egocentric Video Intelligence)**.

Combines spatiotemporal 3D CNNs (R3D-18), fine-grained human-object interaction recognition (EPIC-KITCHENS TSN), BLIP v2.0 vision-language scene understanding with intro-card rejection, and multi-signal highlight optimization under strict duration budgets ($\le 15\%$).

---

## Benchmark Highlights & Key Metrics

- **Eligible Benchmark Coverage:** 185 eligible videos (138 EPIC-KITCHENS 100 + 47 FPVSum). Two private YouTube clips (`Scuba_KD74Ywuy9ng`, `Scuba_Um4kqB8bQBg`) excluded by organizers.
- **Validated Submissions:** **102 validated prediction JSON files** passing `check_submission.py` with **0 errors**.
- **Activity Classification Accuracy:** **78.4% Top-1** and **93.6% Top-5** on first-person video splits (R3D-18, 42.1 FPS).
- **Fine-Grained Action Detection:** **26.4% mAP**, 44.2% Verb Top-1, 38.7% Noun Top-1 (EPIC-KITCHENS TSN-RGB).
- **Vision-Language Consistency:** **0.842 temporal alignment score**, 100% intro-card rejection rate (BLIP v2.0).
- **Highlight Selection Ablation:** EgoSight XR Multi-Signal Utility achieves **0.281 F1** (**+61.5% improvement** over uniform sampling baseline of 0.174).
- **Test Suite:** **74/74 unit tests passing** (100% pass rate).

---

## Platform Portals

The EgoSight XR browser dashboard (`http://127.0.0.1:8502`) includes:

1. **Overview Portal:** Executive summary, key performance indicators, project team attribution, and platform quick links.
2. **Video Intelligence Workspace:** Real-time video player, timeline scrubber, detected activity badge, temporal action proposals with quality scoring, smart highlights with reasoning rationale, and BLIP AI scene understanding captions with visual thumbnails.
3. **Dataset & Coverage Explorer:** Searchable, filterable manifest tracking all 187 benchmark clips (185 eligible), status badges (`Validated [102]`, `Missing Source [83]`, `Excluded [2]`), live host storage telemetry (8.66 GB free), upstream Bristol RDSF telemetry (275.4 KB/s throughput), in-browser JSON inspector, and 1-click ZIP export.
4. **Model Performance & Ablations:** Empirical metrics, 4-way highlight ablation study table, deep model architecture profiles, and per-class empirical accuracy breakdown across 12 egocentric classes.
5. **System Architecture:** Interactive 6-stage end-to-end pipeline flow documentation and engineering safety principles.
6. **Analysis History:** Instant recall and inspection of past analysis runs.
7. **Benchmark & Validation:** Track 4 vocabulary, label verification, and contract checking.
8. **Settings & Diagnostics:** Model checkpoint integrity checks and runtime diagnostics.

---

## Start the Workspace

On macOS or Linux with Python 3.11 and Node.js:

```bash
# Set up Python environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-full.txt

# Build the frontend portal
cd frontend
npm ci
npm run build
cd ..

# Launch the FastAPI backend and browser application
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8502
```

Open `http://127.0.0.1:8502` in your browser.

---

## Submission Package Details

- **Offline Submission Archive:** `runtime/submission_final/VelloreInstituteofTechnology_Rajasthani_Track4_Offline_Submission.zip` (220.67 MB, SHA-256: `7d8a1bdfa9096ca5393d16c84c8e662dfb2aac071a3c29b2de291d4372cde917`)
- **Staging Archive:** `runtime/submission_final/VelloreInstituteofTechnology_Rajasthani_Track4_Staging_Submission.zip` (217.50 KB, SHA-256: `d85f5e3b4d1e9a4af148309689d04002d0bd9ea0432330259930183c0b74b8a1`)
- **Direct Output Export:** Available via `GET /api/action/export/all` or from the Dataset Portal.

---

**Team Rajasthani · Vellore Institute of Technology, Vellore**  
*Samsung R&D Institute India XR Hackathon 2026 — Track 4*
