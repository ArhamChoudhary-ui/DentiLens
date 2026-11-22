"""AI Scene Understanding & Temporal Visual Narration for EgoSight XR.

Grounds egocentric video analysis with timestamped visual descriptions
generated from sampled video frames using Salesforce/blip-image-captioning-base.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import cv2
from PIL import Image

LOG = logging.getLogger(__name__)

MODEL_ID = "Salesforce/blip-image-captioning-base"


class BlipCaptioner:
    """Lazy-loaded singleton for BLIP image captioning on Apple Silicon MPS/CPU."""

    _instance: Optional[BlipCaptioner] = None

    def __init__(self):
        self._processor = None
        self._model = None
        self._device = None

    @classmethod
    def get_instance(cls) -> BlipCaptioner:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        if self._model is not None:
            return

        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        LOG.info("Loading BLIP image captioning model (%s) on %s...", MODEL_ID, self._device)
        self._processor = BlipProcessor.from_pretrained(MODEL_ID)
        try:
            self._model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(self._device)
        except Exception as exc:
            if self._device != "cpu":
                LOG.warning("Failed to load on %s (%s); falling back to CPU", self._device, exc)
                self._device = "cpu"
                self._model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to("cpu")
            else:
                raise

    @property
    def device(self) -> str:
        self._load()
        return self._device or "cpu"

    def caption_image(self, image: Image.Image) -> str:
        """Generate a natural visual caption for a single PIL image."""
        self._load()
        import torch

        try:
            inputs = self._processor(image, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=35,
                    num_beams=3,
                    min_length=5,
                    repetition_penalty=1.1,
                )
            text = self._processor.decode(outputs[0], skip_special_tokens=True).strip()
            return self._clean_caption(text)
        except Exception as exc:
            # Fallback to CPU once if MPS generates a transient error
            if self._device != "cpu":
                LOG.warning("MPS generation failed (%s); retrying on CPU", exc)
                self._device = "cpu"
                self._model = self._model.to("cpu")
                inputs = self._processor(image, return_tensors="pt").to("cpu")
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=35,
                        num_beams=3,
                        min_length=5,
                    )
                text = self._processor.decode(outputs[0], skip_special_tokens=True).strip()
                return self._clean_caption(text)
            raise

    @staticmethod
    def _clean_caption(caption: str) -> str:
        if not caption:
            return "Visual scene content detected."
        # Capitalize first letter
        caption = caption[0].upper() + caption[1:]
        # Ensure punctuation
        if not re.search(r"[.!?]$", caption):
            caption += "."
        return caption


def sample_scene_timestamps(
    duration: float,
    highlights: Optional[List[Dict[str, Any]]] = None,
    actions: Optional[List[Dict[str, Any]]] = None,
    target_count: int = 14,
    min_gap: float = 2.0,
) -> List[Dict[str, Any]]:
    """Determine representative, chronologically organized timestamps to caption.

    Samples frames around:
    1. Highlight intervals (primary importance)
    2. Selected action proposals
    3. Uniform coverage across video start, middle, and end.
    """
    if duration <= 0:
        return []

    candidates: List[Dict[str, Any]] = []

    # 1. Highlights
    if highlights:
        for idx, hl in enumerate(highlights):
            seg = hl.get("segment") or [0, 0]
            s, e = float(seg[0]), float(seg[1])
            mid = (s + e) / 2.0
            candidates.append({
                "timestamp": mid,
                "priority": 1,
                "context": f"Highlight moment #{idx + 1}",
            })
            if (e - s) >= 5.0:
                candidates.append({
                    "timestamp": min(e - 0.5, s + 1.5),
                    "priority": 2,
                    "context": f"Highlight moment #{idx + 1} (action)",
                })

    # 2. Action proposals
    if actions:
        sorted_actions = sorted(actions, key=lambda a: float(a.get("score", 0.0)), reverse=True)
        for idx, act in enumerate(sorted_actions[:8]):
            seg = act.get("segment") or [0, 0]
            mid = (float(seg[0]) + float(seg[1])) / 2.0
            candidates.append({
                "timestamp": mid,
                "priority": 3,
                "context": "Action interval",
            })

    # 3. Uniform coverage across timeline
    steps = [0.03, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 0.97]
    for step in steps:
        ts = step * duration
        candidates.append({
            "timestamp": ts,
            "priority": 4,
            "context": "Scene overview",
        })

    # Filter within valid boundaries
    valid = []
    for c in candidates:
        ts = max(0.2, min(duration - 0.2, c["timestamp"]))
        valid.append({**c, "timestamp": ts})

    # Sort by priority to select the best candidates
    valid.sort(key=lambda c: (c["priority"], c["timestamp"]))

    # Deduplicate within min_gap
    selected: List[Dict[str, Any]] = []
    for c in valid:
        if any(abs(c["timestamp"] - s["timestamp"]) < min_gap for s in selected):
            continue
        selected.append(c)
        if len(selected) >= target_count:
            break

    # Sort chronologically
    selected.sort(key=lambda c: c["timestamp"])
    return selected


def format_clock(seconds: float) -> str:
    """Format seconds into MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def analyze_scene_understanding(
    video_path: Path,
    highlights: Optional[List[Dict[str, Any]]] = None,
    actions: Optional[List[Dict[str, Any]]] = None,
    duration: float = 0.0,
    output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    max_frames: int = 14,
) -> Dict[str, Any]:
    """Extract sampled frames, generate BLIP captions, and save thumbnails and cache.

    Results are persisted to `output_dir / 'scene_understanding.json'`.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        return {
            "status": "failed",
            "model": MODEL_ID,
            "error": f"Video file not found: {video_path.name}",
            "scenes": [],
        }

    # Check cache if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        cache_path = output_dir / "scene_understanding.json"
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text())
                if cached.get("status") == "ready" and cached.get("scenes"):
                    LOG.info("Loaded cached scene understanding for %s", video_path.name)
                    return cached
            except Exception as exc:
                LOG.warning("Could not read cached scene understanding: %s", exc)

    # Determine duration if not provided
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "status": "failed",
            "model": MODEL_ID,
            "error": "Could not open video for scene understanding.",
            "scenes": [],
        }

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if duration <= 0 and total_frames > 0 and fps > 0:
            duration = total_frames / fps

        # Sample timestamps
        targets = sample_scene_timestamps(
            duration=duration,
            highlights=highlights,
            actions=actions,
            target_count=max_frames,
            min_gap=max(1.8, duration / 25.0),
        )

        if not targets:
            return {
                "status": "ready",
                "model": MODEL_ID,
                "scenes": [],
                "frame_count": 0,
            }

        # Setup thumbnails directory
        thumb_dir = None
        if output_dir:
            thumb_dir = output_dir / "thumbnails"
            thumb_dir.mkdir(parents=True, exist_ok=True)

        captioner = BlipCaptioner.get_instance()
        scenes = []

        for idx, target in enumerate(targets):
            ts = target["timestamp"]
            frame_idx = int(round(ts * fps))
            if total_frames > 0:
                frame_idx = min(int(total_frames - 1), max(0, frame_idx))

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Generate caption
            try:
                caption = captioner.caption_image(pil_img)
            except Exception as exc:
                LOG.warning("BLIP failed on frame at %.1fs: %s", ts, exc)
                caption = "Visual scene present."

            # Save thumbnail
            thumb_url = None
            if thumb_dir:
                thumb_filename = f"thumb_{idx:02d}_{int(ts * 10):05d}.jpg"
                thumb_file = thumb_dir / thumb_filename
                if not thumb_file.is_file():
                    # Resize to max width 360 maintaining aspect ratio
                    h, w = frame.shape[:2]
                    target_w = 360
                    target_h = int(round(h * (target_w / w)))
                    resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(thumb_file), resized, [cv2.IMWRITE_JPEG_QUALITY, 82])

                if run_id:
                    thumb_url = f"/api/action/runs/{run_id}/thumbnails/{thumb_filename}"
                else:
                    thumb_url = f"thumbnails/{thumb_filename}"

            scenes.append({
                "index": idx,
                "timestamp": round(ts, 2),
                "timestamp_formatted": format_clock(ts),
                "caption": caption,
                "context": target.get("context", "Scene observation"),
                "thumbnail": thumb_url,
                "model": MODEL_ID,
            })

        result = {
            "status": "ready",
            "model": MODEL_ID,
            "device": captioner.device,
            "frame_count": len(scenes),
            "scenes": scenes,
        }

        # Persist cache
        if output_dir:
            cache_file = output_dir / "scene_understanding.json"
            cache_file.write_text(json.dumps(result, indent=2))

        return result

    except Exception as exc:
        LOG.exception("Scene understanding analysis failed: %s", exc)
        return {
            "status": "failed",
            "model": MODEL_ID,
            "error": str(exc),
            "scenes": [],
        }
    finally:
        cap.release()
