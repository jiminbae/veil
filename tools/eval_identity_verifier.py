#!/usr/bin/env python3
"""Evaluate identity preservation/leakage with a pluggable verifier backend.

This script is intentionally separate from VEIL's internal target-matching cache.
The built-in debug_color backend is for pipeline smoke tests only; it is not a
paper-valid face-recognition verifier. Plug in an independent backend such as
AdaFace, MagFace, FaceNet, or a custom ONNX model for paper results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - cv2 is expected in the VEIL env.
    cv2 = None


ACTIONS_FOR_NON_TARGET = {"SWAP", "BLUR", "UNPROCESSED", "FAILED", "UNKNOWN"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class VerifierError(RuntimeError):
    """Raised when verifier evaluation cannot proceed."""


class Verifier:
    """Clean verifier interface for independent face-recognition backends."""

    name = "base"
    paper_valid = False

    def extract_embedding(self, image: Image.Image) -> np.ndarray | None:
        raise NotImplementedError

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return float("nan")
        return float(np.dot(a, b) / denom)


class DebugColorVerifier(Verifier):
    """Small deterministic smoke-test backend; not a face-recognition model."""

    name = "debug_color"
    paper_valid = False

    def extract_embedding(self, image: Image.Image) -> np.ndarray | None:
        resized = image.convert("RGB").resize((32, 32))
        arr = np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0
        arr = arr - float(arr.mean())
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            return None
        return arr / norm


class UnimplementedVerifier(Verifier):
    def __init__(self, backend: str, model_path: Path | None = None):
        self.name = backend
        self.model_path = model_path

    def extract_embedding(self, image: Image.Image) -> np.ndarray | None:
        hint = ""
        if self.model_path is not None:
            hint = f" configured at {self.model_path}"
        raise VerifierError(
            f"Verifier backend '{self.name}'{hint} is not implemented in this repo yet. "
            "Add an independent AdaFace, MagFace, FaceNet, or custom ONNX adapter. "
            "Do not use VEIL's internal InsightFace buffalo_l checkpoint as the verifier."
        )


def build_verifier(backend: str, model_path: Path | None = None) -> Verifier:
    if backend == "debug_color":
        return DebugColorVerifier()
    if backend in {'adaface', 'facenet'}:
        from tools.independent_verifiers import FaceRecognitionVerifier
        try:
            return FaceRecognitionVerifier(backend, model_path)
        except (ValueError, FileNotFoundError) as exc:
            raise VerifierError(str(exc)) from exc
    if backend in {"magface", "custom_onnx"}:
        return UnimplementedVerifier(backend, model_path)
    raise VerifierError(f"Unknown verifier backend: {backend}")


class FrameSource:
    def get_frame(self, frame_idx: int) -> Image.Image | None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class DirectoryFrameSource(FrameSource):
    def __init__(self, path: Path):
        self.path = path
        self.by_index: dict[int, Path] = {}
        for image_path in sorted(path.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            digits = "".join(ch for ch in image_path.stem if ch.isdigit())
            if digits:
                self.by_index.setdefault(int(digits), image_path)
        self.images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

    def get_frame(self, frame_idx: int) -> Image.Image | None:
        path = self.by_index.get(frame_idx)
        if path is None and 1 <= frame_idx <= len(self.images):
            path = self.images[frame_idx - 1]
        if path is None:
            return None
        return Image.open(path).convert("RGB")


class VideoFrameSource(FrameSource):
    def __init__(self, path: Path):
        if cv2 is None:
            raise VerifierError("OpenCV is required to read video inputs")
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        self._last_frame_idx: int | None = None
        self._last_frame: Image.Image | None = None
        if not self.cap.isOpened():
            raise VerifierError(f"Cannot open video: {path}")

    def get_frame(self, frame_idx: int) -> Image.Image | None:
        frame_idx = int(frame_idx)
        if self._last_frame_idx == frame_idx and self._last_frame is not None:
            return self._last_frame
        zero_based = max(frame_idx - 1, 0)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, zero_based)
        ok, frame = self.cap.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._last_frame_idx = frame_idx
        self._last_frame = Image.fromarray(rgb)
        return self._last_frame

    def close(self) -> None:
        self.cap.release()


def open_frame_source(path: Path) -> FrameSource:
    path = path.expanduser().resolve()
    if path.is_dir():
        return DirectoryFrameSource(path)
    if path.is_file():
        return VideoFrameSource(path)
    raise VerifierError(f"Frame source does not exist: {path}")


@dataclass
class PairSpec:
    pair_type: str
    frame_idx: int
    original_bbox: list[float] | None
    processed_bbox: list[float]
    identity_id: str
    action: str
    box_source: str


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VerifierError(f"Invalid JSON: {path}") from exc


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_bbox(value) -> list[float] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            value = json.loads(text)
        else:
            value = [float(part) for part in text.replace(";", ",").split(",")]
    if not isinstance(value, Sequence) or len(value) != 4:
        raise VerifierError(f"Invalid bbox: {value}")
    return [float(v) for v in value]


def clamp_bbox(bbox: Sequence[float], size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    width, height = size
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, int(round(x1))))
    x2 = max(0, min(width, int(round(x2))))
    y1 = max(0, min(height, int(round(y1))))
    y2 = max(0, min(height, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_image(image: Image.Image, bbox: Sequence[float]) -> Image.Image | None:
    clipped = clamp_bbox(bbox, image.size)
    if clipped is None:
        return None
    return image.crop(clipped)


def row_is_target(row: dict) -> bool:
    return bool(row.get("is_target_final", row.get("is_target", False)))


def row_action(row: dict) -> str:
    action = row.get("final_action")
    return str(action).upper() if action else "UNKNOWN"


def load_metadata_pairs(metadata_path: Path) -> list[PairSpec]:
    rows = read_json(metadata_path)
    if not isinstance(rows, list):
        raise VerifierError(f"Expected list action metadata: {metadata_path}")
    pairs: list[PairSpec] = []
    for row in rows:
        bbox = parse_bbox(row.get("bbox") or row.get("smoothed_bbox"))
        if bbox is None:
            continue
        frame_idx = int(row.get("frame_idx", row.get("frame_number", 0)))
        if frame_idx <= 0:
            continue
        identity = row.get("stable_face_id", row.get("raw_track_id", "unknown"))
        action = row_action(row)
        if row_is_target(row):
            pairs.append(
                PairSpec(
                    pair_type="target",
                    frame_idx=frame_idx,
                    original_bbox=None,
                    processed_bbox=bbox,
                    identity_id=str(identity),
                    action=action,
                    box_source="metadata",
                )
            )
        elif bool(row.get("is_background", True)) and action in ACTIONS_FOR_NON_TARGET:
            pairs.append(
                PairSpec(
                    pair_type="non_target",
                    frame_idx=frame_idx,
                    original_bbox=bbox,
                    processed_bbox=bbox,
                    identity_id=str(identity),
                    action=action,
                    box_source="metadata",
                )
            )
    return pairs


def load_gt_pairs(gt_path: Path) -> list[PairSpec]:
    if gt_path.suffix.lower() == ".csv":
        with gt_path.open("r", encoding="utf-8", newline="") as stream:
            records = list(csv.DictReader(stream))
    else:
        payload = read_json(gt_path)
        if isinstance(payload, dict):
            records = payload.get("pairs") or payload.get("annotations") or []
        else:
            records = payload
    pairs: list[PairSpec] = []
    for idx, record in enumerate(records):
        pair_type = str(record.get("pair_type", record.get("type", record.get("label", "")))).lower()
        if pair_type in {"protected", "target"}:
            pair_type = "target"
        elif pair_type in {"background", "non_target", "nontarget"}:
            pair_type = "non_target"
        else:
            raise VerifierError(f"GT record {idx} has invalid pair type: {pair_type}")
        frame_idx = int(record.get("frame_idx", record.get("frame", 0)))
        processed_bbox = parse_bbox(record.get("processed_bbox", record.get("bbox")))
        original_bbox = parse_bbox(record.get("original_bbox", record.get("bbox")))
        if processed_bbox is None:
            raise VerifierError(f"GT record {idx} is missing processed bbox")
        pairs.append(
            PairSpec(
                pair_type=pair_type,
                frame_idx=frame_idx,
                original_bbox=original_bbox,
                processed_bbox=processed_bbox,
                identity_id=str(record.get("identity_id", record.get("id", idx))),
                action=str(record.get("action", "GT")).upper(),
                box_source="gt",
            )
        )
    return pairs


def summarize_scores(scores: Sequence[float], threshold: float) -> dict[str, float | int]:
    valid = [score for score in scores if score is not None and math.isfinite(score)]
    return {
        "pairs": len(valid),
        "mean": round(float(statistics.fmean(valid)), 6) if valid else None,
        "median": round(float(statistics.median(valid)), 6) if valid else None,
        "accept_rate": round(sum(1 for score in valid if score >= threshold) / len(valid), 6) if valid else None,
    }


def save_debug_pair(
    output_dir: Path,
    category: str,
    row: dict,
    images: Sequence[tuple[str, Image.Image]],
) -> None:
    safe_id = f"{row['pair_type']}_f{row['frame_idx']:06d}_{row['identity_id']}_{row['score']:.3f}"
    safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in safe_id)
    target_dir = output_dir / "debug_crops" / category / safe_id
    target_dir.mkdir(parents=True, exist_ok=True)
    write_json(target_dir / "metadata.json", row)
    for name, image in images:
        image.save(target_dir / f"{name}.jpg", quality=92)


def evaluate_identity(
    *,
    original: Path,
    processed: Path,
    action_metadata: Path,
    protected_reference: Path,
    output_dir: Path,
    backend: str = "debug_color",
    verifier_model: Path | None = None,
    gt_annotations: Path | None = None,
    box_source: str = "metadata",
    condition: str = "unknown",
    clip_id: str = "unknown",
    threshold: float = 0.5,
    max_debug_crops: int = 5,
    max_pairs: int | None = None,
) -> dict:
    verifier = build_verifier(backend, verifier_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_source = open_frame_source(original)
    processed_source = open_frame_source(processed)
    try:
        reference = Image.open(protected_reference).convert("RGB")
        reference_embedding = verifier.extract_embedding(reference)
        if reference_embedding is None:
            raise VerifierError(f"Could not extract reference embedding: {protected_reference}")

        if box_source == "gt":
            if gt_annotations is None:
                raise VerifierError("--box-source gt requires --gt-annotations")
            pairs = load_gt_pairs(gt_annotations)
        elif box_source == "metadata":
            pairs = load_metadata_pairs(action_metadata)
        else:
            raise VerifierError(f"Unknown box source: {box_source}")
        if max_pairs is not None:
            pairs = pairs[:max_pairs]

        detailed_rows: list[dict] = []
        target_low_risk: list[tuple[float, dict, list[tuple[str, Image.Image]]]] = []
        non_target_high_risk: list[tuple[float, dict, list[tuple[str, Image.Image]]]] = []

        for pair in pairs:
            row = {
                'condition':condition,'clip_id':clip_id,'pair_type':pair.pair_type,
                'frame_idx':pair.frame_idx,'identity_id':pair.identity_id,'action':pair.action,
                'box_source':pair.box_source,'verifier_backend':verifier.name,'score':None,
                'threshold':threshold,'accepted':None,'paper_valid_backend':int(verifier.paper_valid),
                'status':'frame_or_embedding_unavailable',
            }
            detailed_rows.append(row)
            processed_frame = processed_source.get_frame(pair.frame_idx)
            if processed_frame is None:
                continue
            processed_crop = crop_image(processed_frame, pair.processed_bbox)
            if processed_crop is None:
                continue

            if pair.pair_type == "target":
                left_embedding = reference_embedding
                right_embedding = verifier.extract_embedding(processed_crop)
                score_images = [("reference", reference), ("processed", processed_crop)]
            else:
                original_frame = original_source.get_frame(pair.frame_idx)
                if original_frame is None or pair.original_bbox is None:
                    continue
                original_crop = crop_image(original_frame, pair.original_bbox)
                if original_crop is None:
                    continue
                left_embedding = verifier.extract_embedding(original_crop)
                right_embedding = verifier.extract_embedding(processed_crop)
                score_images = [("original", original_crop), ("processed", processed_crop)]

            if left_embedding is None or right_embedding is None:
                continue
            score = verifier.similarity(left_embedding, right_embedding)
            row.update({
                "condition": condition,
                "clip_id": clip_id,
                "pair_type": pair.pair_type,
                "frame_idx": pair.frame_idx,
                "identity_id": pair.identity_id,
                "action": pair.action,
                "box_source": pair.box_source,
                "verifier_backend": verifier.name,
                "score": round(float(score), 6),
                "threshold": threshold,
                "accepted": int(score >= threshold),
                "paper_valid_backend": int(verifier.paper_valid),
                'status':'scored',
            })
            if pair.pair_type == "target" and score < threshold:
                target_low_risk.append((score, row, score_images))
            if pair.pair_type == "non_target" and score >= threshold:
                non_target_high_risk.append((score, row, score_images))

        target_scores = [row["score"] for row in detailed_rows if row["pair_type"] == "target"]
        non_target_scores = [row["score"] for row in detailed_rows if row["pair_type"] == "non_target"]
        target_summary = summarize_scores(target_scores, threshold)
        non_target_summary = summarize_scores(non_target_scores, threshold)
        per_clip_row = {
            "condition": condition,
            "clip_id": clip_id,
            "verifier_backend": verifier.name,
            "paper_valid_backend": int(verifier.paper_valid),
            "box_source": box_source,
            "threshold": threshold,
            'target_attempts':sum(p.pair_type=='target' for p in pairs),
            'non_target_attempts':sum(p.pair_type=='non_target' for p in pairs),
            'extraction_failures':sum(r['score'] is None for r in detailed_rows),
            "target_pairs": target_summary["pairs"],
            "target_mean": target_summary["mean"],
            "target_median": target_summary["median"],
            "target_accept_rate": target_summary["accept_rate"],
            "non_target_pairs": non_target_summary["pairs"],
            "non_target_mean": non_target_summary["mean"],
            "non_target_median": non_target_summary["median"],
            "non_target_accept_rate": non_target_summary["accept_rate"],
        }

        fields = [
            "condition",
            "clip_id",
            "pair_type",
            "frame_idx",
            "identity_id",
            "action",
            "box_source",
            "verifier_backend",
            "score",
            "threshold",
            "accepted",
            "paper_valid_backend",
            'status',
        ]
        write_csv(output_dir / "pair_scores.csv", detailed_rows, fields)
        write_csv(output_dir / "per_clip_identity_verifier.csv", [per_clip_row])
        write_csv(output_dir / "aggregate_identity_verifier.csv", [per_clip_row])
        write_json(
            output_dir / "evaluation_config.json",
            {
                "backend": backend,
                "paper_valid_backend": verifier.paper_valid,
                "backend_note": "debug_color is for smoke tests only" if backend == "debug_color" else "independent verifier backend",
                "box_source": box_source,
                "condition": condition,
                "clip_id": clip_id,
                "threshold": threshold,
                'threshold_calibrated':False,
                'threshold_note':'CLI threshold; external calibration must be supplied and documented by caller. Prefer score_research_identity for calibrated matrix evaluation.',
                "max_pairs": max_pairs,
                "gt_annotations": str(gt_annotations) if gt_annotations else None,
                "action_metadata": str(action_metadata),
                "protected_reference": str(protected_reference),
            },
        )

        for _, row, images in sorted(non_target_high_risk, key=lambda item: item[0], reverse=True)[:max_debug_crops]:
            save_debug_pair(output_dir, "high_non_target_similarity", row, images)
        for _, row, images in sorted(target_low_risk, key=lambda item: item[0])[:max_debug_crops]:
            save_debug_pair(output_dir, "low_target_similarity", row, images)

        return {
            "output_dir": str(output_dir),
            "pair_scores_csv": str(output_dir / "pair_scores.csv"),
            "per_clip_csv": str(output_dir / "per_clip_identity_verifier.csv"),
            "aggregate_csv": str(output_dir / "aggregate_identity_verifier.csv"),
            **per_clip_row,
        }
    finally:
        original_source.close()
        processed_source.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate identity preservation/leakage with a pluggable verifier.")
    parser.add_argument("--original", type=Path, required=True, help="Original video path or frame directory.")
    parser.add_argument("--processed", type=Path, required=True, help="Processed video path or frame directory.")
    parser.add_argument("--action-metadata", type=Path, required=True, help="VEIL face/action metadata JSON.")
    parser.add_argument("--protected-reference", type=Path, required=True, help="Protected reference image.")
    parser.add_argument("--gt-annotations", type=Path, help="Optional GT pair annotations JSON/CSV.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", default="unknown")
    parser.add_argument("--clip-id", default="unknown")
    parser.add_argument("--box-source", choices=["metadata", "gt"], default="metadata")
    parser.add_argument("--backend", choices=["debug_color", "adaface", "magface", "facenet", "custom_onnx"], default="debug_color")
    parser.add_argument("--verifier-model", type=Path, help="Optional independent verifier model/checkpoint path.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-debug-crops", type=int, default=5)
    parser.add_argument("--max-pairs", type=int, help="Optional cap for quick smoke tests; omit for full evaluation.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = evaluate_identity(
            original=args.original,
            processed=args.processed,
            action_metadata=args.action_metadata,
            protected_reference=args.protected_reference,
            output_dir=args.output_dir,
            backend=args.backend,
            verifier_model=args.verifier_model,
            gt_annotations=args.gt_annotations,
            box_source=args.box_source,
            condition=args.condition,
            clip_id=args.clip_id,
            threshold=args.threshold,
            max_debug_crops=args.max_debug_crops,
            max_pairs=args.max_pairs,
        )
    except VerifierError as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
