#!/usr/bin/env python3
"""Evaluate VEIL face metadata without assuming unavailable action fields."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


STATES = ("PRESERVE", "SWAP", "BLUR", "UNPROCESSED", "FAILED", "UNKNOWN")
TRACK_LOG_RE = re.compile(
    r"Frame=(?P<frame>\d+).*?TrackID=(?P<track>\d+).*?SwapSuccess=(?P<swap>True|False)"
)


class EvaluationError(RuntimeError):
    """Raised when VEIL metadata cannot be evaluated."""


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Invalid JSON: {path}") from exc


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def find_run_dirs(runs_dir: Path) -> list[Path]:
    runs_dir = runs_dir.expanduser().resolve()
    if not runs_dir.exists():
        raise EvaluationError(f"Runs directory does not exist: {runs_dir}")
    run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    if not run_dirs:
        raise EvaluationError(f"No run directories found under: {runs_dir}")
    return run_dirs


def find_one(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[0] if matches else None


def load_review(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    path = path.expanduser().resolve()
    if not path.exists():
        raise EvaluationError(f"Review CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {row["clip_id"]: row for row in csv.DictReader(stream)}


def schema_report(face_paths: Sequence[Path], tracking_paths: Sequence[Path]) -> dict:
    file_reports: list[dict] = []
    all_fields: list[set[str]] = []
    examples: dict[str, object] = {}

    for path in face_paths:
        rows = read_json(path)
        if not isinstance(rows, list):
            raise EvaluationError(f"Expected list metadata: {path}")
        fields = set()
        for row in rows:
            fields.update(row)
            for key, value in row.items():
                examples.setdefault(key, value)
        all_fields.append(fields)
        file_reports.append({"path": str(path), "rows": len(rows), "fields": sorted(fields)})

    common_fields = sorted(set.intersection(*all_fields)) if all_fields else []
    union_fields = sorted(set.union(*all_fields)) if all_fields else []
    optional_fields = sorted(set(union_fields) - set(common_fields))

    tracking_reports: list[dict] = []
    for path in tracking_paths:
        rows = read_json(path)
        fields = sorted({key for row in rows for key in row}) if isinstance(rows, list) else []
        tracking_reports.append({"path": str(path), "rows": len(rows) if isinstance(rows, list) else None, "fields": fields})

    return {
        "face_metadata_files": len(face_paths),
        "tracking_metadata_files": len(tracking_paths),
        "common_fields": common_fields,
        "optional_fields": optional_fields,
        "file_reports": file_reports,
        "tracking_file_reports": tracking_reports,
        "example_values": {key: examples[key] for key in sorted(examples)},
        "action_field_present": any(field in union_fields for field in ("action", "processing_action", "final_action")),
        "swap_success_field_present": "swap_success" in union_fields,
    }


def parse_bbox(value) -> list[float] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            value = json.loads(value)
        else:
            value = [float(part) for part in value.replace(";", ",").split(",")]
    if not isinstance(value, Sequence) or len(value) != 4:
        raise EvaluationError(f"Invalid bbox: {value}")
    return [float(v) for v in value]


def load_gt_annotations(path: Path | None) -> dict[str, list[dict]]:
    if path is None:
        return {}
    path = path.expanduser().resolve()
    if not path.exists():
        raise EvaluationError(f"GT annotation path does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            records = list(csv.DictReader(stream))
    elif path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = read_json(path)
        records = payload.get("annotations", payload.get("targets", [])) if isinstance(payload, dict) else payload

    by_clip: dict[str, list[dict]] = defaultdict(list)
    for idx, record in enumerate(records):
        clip_id = str(record.get("clip_id", "")).strip()
        if not clip_id:
            raise EvaluationError(f"GT annotation row {idx} is missing clip_id")
        bbox = parse_bbox(record.get("bbox"))
        if bbox is None:
            try:
                bbox = [
                    float(record["bbox_x1"]),
                    float(record["bbox_y1"]),
                    float(record["bbox_x2"]),
                    float(record["bbox_y2"]),
                ]
            except KeyError as exc:
                raise EvaluationError(f"GT annotation row {idx} is missing bbox fields") from exc
        row = {
            "clip_id": clip_id,
            "frame_idx": int(record.get("frame_idx", record.get("frame", 0))),
            "person_id": str(record.get("person_id", record.get("target_id", "target"))),
            "bbox": bbox,
            "visibility": record.get("visibility", ""),
            "notes": record.get("notes", ""),
            "is_target": str(record.get('is_target', True)).lower() in {'true','1','yes'},
            "annotation_source": record.get('annotation_source', 'unspecified'),
            "verification_status": record.get('verification_status', 'unspecified'),
        }
        if row["frame_idx"] <= 0:
            raise EvaluationError(f"GT annotation row {idx} has invalid frame_idx: {row['frame_idx']}")
        by_clip[clip_id].append(row)
    return {clip_id: sorted(rows, key=lambda row: (row["frame_idx"], row["person_id"])) for clip_id, rows in by_clip.items()}


def bbox_iou(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def empty_gt_metrics(iou_threshold: float) -> dict:
    return {
        "gt_iou_threshold": iou_threshold,
        "gt_target_boxes": 0,
        "gt_target_matched_boxes": 0,
        "gt_target_missed_boxes": 0,
        "gt_target_preserve_boxes": 0,
        "gt_target_altered_boxes": 0,
        "gt_target_swap_boxes": 0,
        "gt_target_blur_boxes": 0,
        "gt_target_other_action_boxes": 0,
        "gt_target_match_rate": None,
        "gt_target_processed_as_preserve_rate": None,
        "gt_target_altered_rate": None,
        "gt_target_missed_rate": None,
    }


def compute_gt_target_metrics(
    clip_id: str,
    actions: Sequence[dict],
    gt_rows: Sequence[dict],
    *,
    iou_threshold: float,
) -> tuple[dict, list[dict]]:
    metrics = empty_gt_metrics(iou_threshold)
    if not gt_rows:
        return metrics, []

    by_frame: dict[int, list[dict]] = defaultdict(list)
    for action in actions:
        if action.get("bbox") is not None:
            by_frame[int(action["frame_idx"])].append(action)

    match_rows: list[dict] = []
    counts = Counter()
    # Reserve observations once per frame; one detection cannot cover two people.
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    gt_by_frame = defaultdict(list)
    for gt in gt_rows:
        gt_by_frame[int(gt['frame_idx'])].append(gt)
    assignments = {}
    for frame, rows in gt_by_frame.items():
        candidates = by_frame.get(frame, [])
        if not candidates:
            continue
        ious = np.array([[bbox_iou(g['bbox'], a.get('bbox')) for a in candidates] for g in rows])
        weights = np.where(ious >= iou_threshold, len(rows) + ious, 0.0)
        gi, ai = linear_sum_assignment(-weights)
        for gidx, aidx in zip(gi, ai):
            if ious[gidx, aidx] >= iou_threshold:
                assignments[id(rows[gidx])] = (candidates[aidx], float(ious[gidx, aidx]))
    for gt in gt_rows:
        if not gt.get('is_target', True):
            continue
        candidates = by_frame.get(int(gt["frame_idx"]), [])
        best_action = None
        best_iou = 0.0
        if id(gt) in assignments:
            best_action, best_iou = assignments[id(gt)]

        matched = best_action is not None and best_iou >= iou_threshold
        state = best_action["state"] if matched else "MISSED"
        if matched:
            counts["matched"] += 1
            if state == "PRESERVE":
                counts["preserve"] += 1
            elif state in {"SWAP", "BLUR"}:
                counts["altered"] += 1
                counts[state.lower()] += 1
            else:
                counts["other"] += 1
        else:
            counts["missed"] += 1

        match_rows.append(
            {
                "clip_id": clip_id,
                "frame_idx": gt["frame_idx"],
                "person_id": gt["person_id"],
                "visibility": gt.get("visibility", ""),
                "gt_bbox": json.dumps(gt["bbox"]),
                "matched": int(matched),
                "matched_iou": round(best_iou, 6),
                "matched_state": state,
                "matched_raw_track_id": best_action.get("raw_track_id") if matched else "",
                "matched_stable_face_id": best_action.get("stable_face_id") if matched else "",
                "matched_bbox": json.dumps(best_action.get("bbox")) if matched else "",
                "status": "matched" if matched else "missed_no_observation",
                "notes": gt.get("notes", ""),
            }
        )

    total = sum(bool(g.get('is_target',True)) for g in gt_rows)
    if not total:
        return metrics, match_rows
    metrics.update(
        {
            "gt_target_boxes": total,
            "gt_target_matched_boxes": counts["matched"],
            "gt_target_missed_boxes": counts["missed"],
            "gt_target_preserve_boxes": counts["preserve"],
            "gt_target_altered_boxes": counts["altered"],
            "gt_target_swap_boxes": counts["swap"],
            "gt_target_blur_boxes": counts["blur"],
            "gt_target_other_action_boxes": counts["other"],
            "gt_target_match_rate": round(safe_div(counts["matched"], total), 6),
            "gt_target_processed_as_preserve_rate": round(safe_div(counts["preserve"], total), 6),
            "gt_target_altered_rate": round(safe_div(counts["altered"], total), 6),
            "gt_target_missed_rate": round(safe_div(counts["missed"], total), 6),
        }
    )
    return metrics, match_rows


def parse_track_swap_log(log_path: Path | None) -> dict[tuple[int, int], bool]:
    if log_path is None or not log_path.exists():
        return {}
    result: dict[tuple[int, int], bool] = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TRACK_LOG_RE.search(line)
        if not match:
            continue
        result[(int(match.group("frame")), int(match.group("track")))] = match.group("swap") == "True"
    return result


def dedup_key(row: dict) -> tuple[int, str]:
    frame_idx = int(row.get("frame_idx", row.get("frame_number", -1)))
    stable_face_id = row.get("stable_face_id")
    if stable_face_id is not None:
        return frame_idx, f"face:{stable_face_id}"
    return frame_idx, f"track:{row.get('raw_track_id', row.get('track_id'))}"


def deduplicate_rows(rows: Sequence[dict]) -> list[dict]:
    by_key: dict[tuple[int, str], dict] = {}
    for row in rows:
        key = dedup_key(row)
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = row
            continue
        # Prefer rows with explicit target/background labels and better quality.
        score = (
            int(bool(row.get("is_target"))),
            int(bool(row.get("is_background"))),
            int(row.get("quality") == "GOOD"),
        )
        previous_score = (
            int(bool(previous.get("is_target"))),
            int(bool(previous.get("is_background"))),
            int(previous.get("quality") == "GOOD"),
        )
        if score > previous_score:
            by_key[key] = row
    return [by_key[key] for key in sorted(by_key)]


def classify_row(row: dict, logged_swaps: dict[tuple[int, int], bool], *, blur_fallback_enabled: bool = True) -> tuple[str, str]:
    final_action = row.get("final_action")
    if isinstance(final_action, str):
        normalized = final_action.upper()
        if normalized in STATES:
            return normalized, "metadata_final_action"

    if row.get("is_target") is True:
        return "PRESERVE", "metadata_is_target"

    if row.get("is_background") is True:
        frame_idx = int(row.get("frame_idx", -1))
        raw_track_id = int(row.get("raw_track_id", row.get("track_id", -1)))
        logged = logged_swaps.get((frame_idx, raw_track_id))
        if logged is True:
            return "SWAP", "track_log_swap_success"
        if logged is False:
            if blur_fallback_enabled:
                return "BLUR", "track_log_swap_failed_fallback_blur"
            return "UNPROCESSED", "track_log_swap_failed_no_blur_fallback"

        fallback_reasons = row.get("fallback_reasons") or []
        if row.get("embedding_ok") is False:
            if blur_fallback_enabled:
                return "BLUR", "embedding_failed_fallback_blur"
            return "UNPROCESSED", "embedding_failed_no_blur_fallback"
        if row.get("quality") != "GOOD" or fallback_reasons:
            if blur_fallback_enabled:
                return "BLUR", "quality_or_fallback_blur"
            return "UNPROCESSED", "quality_or_fallback_no_blur_fallback"
        return "UNKNOWN", "background_good_but_swap_not_logged"

    return "UNKNOWN", "not_target_or_background"


def state_switches(actions: Sequence[dict], *, target_only: bool | None) -> int:
    by_identity: dict[str, list[dict]] = defaultdict(list)
    for action in actions:
        if target_only is True and not action["is_target"]:
            continue
        if target_only is False and action["is_target"]:
            continue
        identity = action["dedup_identity"]
        by_identity[identity].append(action)

    switches = 0
    for rows in by_identity.values():
        previous = None
        for row in sorted(rows, key=lambda item: item["frame_idx"]):
            state = row["state"]
            if previous is not None and state != previous:
                switches += 1
            previous = state
    return switches


def safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def evaluate_clip(clip_id: str, run_dir: Path, review: dict | None = None, *, blur_fallback_enabled: bool = True) -> tuple[dict, list[dict], list[dict]]:
    face_path = find_one(run_dir, "face_metadata*.json")
    if face_path is None:
        raise EvaluationError(f"No face_metadata*.json under: {run_dir}")
    rows = read_json(face_path)
    if not isinstance(rows, list):
        raise EvaluationError(f"Expected list metadata: {face_path}")

    log_path = find_one(run_dir, "tracking_target*_log.txt")
    logged_swaps = parse_track_swap_log(log_path)
    deduped = deduplicate_rows(rows)
    max_frame = max((int(row.get("frame_idx", 0)) for row in deduped), default=0)
    frame_ledger = run_dir / 'frames.json'
    denominator_source = 'legacy_last_face_observation; trailing empty frames unverifiable'
    if frame_ledger.exists():
        frames = read_json(frame_ledger)
        max_frame = max((int(row['frame_idx']) for row in frames),default=0)
        denominator_source = 'decoded_frame_ledger'
    actions: list[dict] = []
    for row in deduped:
        state, reason = classify_row(row, logged_swaps, blur_fallback_enabled=blur_fallback_enabled)
        frame_idx, identity = dedup_key(row)
        action = {
            "clip_id": clip_id,
            "frame_idx": frame_idx,
            "raw_track_id": row.get("raw_track_id"),
            "stable_face_id": row.get("stable_face_id"),
            "dedup_identity": identity,
            "is_target": bool(row.get("is_target_final", row.get("is_target"))),
            "is_target_direct": bool(row.get("is_target_direct", row.get("is_target"))),
            "is_target_final": bool(row.get("is_target_final", row.get("is_target"))),
            "is_background": bool(row.get("is_background")),
            "state": state,
            "state_reason": reason,
            "final_action": row.get("final_action"),
            "swap_success": row.get("swap_success"),
            "blur_applied": row.get("blur_applied"),
            "quality": row.get("quality"),
            "fallback_reasons": row.get("fallback_reasons") or [],
            "embedding_ok": row.get("embedding_ok"),
            "target_similarity": row.get("target_similarity"),
            "bbox": row.get("bbox"),
        }
        actions.append(action)

    state_counts = Counter(action["state"] for action in actions)
    protected = [action for action in actions if action["is_target"]]
    non_target = [action for action in actions if action["is_background"]]
    protected_counts = Counter(action["state"] for action in protected)
    non_target_counts = Counter(action["state"] for action in non_target)
    unknown_rows = state_counts["UNKNOWN"]
    protected_denominator = len(protected)
    non_target_denominator = len(non_target)
    protected_frames = {action["frame_idx"] for action in protected}

    metrics = {
        "clip_id": clip_id,
        "accepted": (review or {}).get("accepted", ""),
        "total_frames": max_frame,
        "frame_denominator_source": denominator_source,
        "metadata_rows": len(rows),
        "deduped_face_rows": len(deduped),
        "duplicate_rows_removed": len(rows) - len(deduped),
        "frames_with_any_face": len({action["frame_idx"] for action in actions}),
        "protected_face_rows": len(protected),
        "non_target_face_rows": len(non_target),
        "protected_state_switches": state_switches(actions, target_only=True),
        "non_target_state_switches": state_switches(actions, target_only=False),
        "stable_id_count": len({action["stable_face_id"] for action in actions if action["stable_face_id"] is not None}),
        "raw_track_id_count": len({action["raw_track_id"] for action in actions if action["raw_track_id"] is not None}),
        "target_coverage": round(safe_div(len(protected_frames), max_frame), 6),
        "target_frame_count": len(protected_frames),
        "protected_alteration_rate": round(
            safe_div(protected_counts["SWAP"] + protected_counts["BLUR"] + protected_counts["FAILED"], protected_denominator),
            6,
        ),
        "non_target_exposure_rate": round(
            safe_div(non_target_counts["PRESERVE"] + non_target_counts["UNPROCESSED"], non_target_denominator),
            6,
        ),
        "anonymization_coverage": round(
            safe_div(non_target_counts["SWAP"] + non_target_counts["BLUR"], non_target_denominator),
            6,
        ),
        "swap_coverage": round(safe_div(non_target_counts["SWAP"], non_target_denominator), 6),
        "blur_coverage": round(safe_div(non_target_counts["BLUR"], non_target_denominator), 6),
        "non_target_unprocessed_rate": round(safe_div(non_target_counts["UNPROCESSED"], non_target_denominator), 6),
        "non_target_unknown_rate": round(safe_div(non_target_counts["UNKNOWN"], non_target_denominator), 6),
        "protected_unknown_rate": round(safe_div(protected_counts["UNKNOWN"], protected_denominator), 6),
        "unknown_rate": round(safe_div(unknown_rows, len(actions)), 6),
        "log_path": str(log_path) if log_path else "",
        "face_metadata_path": str(face_path),
    }
    for prefix, counts in (("protected", protected_counts), ("non_target", non_target_counts)):
        for state in STATES:
            metrics[f"{prefix}_{state.lower()}_rows"] = counts[state]
    for state in STATES:
        metrics[f"{state.lower()}_rows"] = state_counts[state]

    identity_rows: list[dict] = []
    by_identity: dict[str, list[dict]] = defaultdict(list)
    for action in actions:
        by_identity[action["dedup_identity"]].append(action)
    for identity, group in sorted(by_identity.items()):
        counts = Counter(action["state"] for action in group)
        identity_rows.append(
            {
                "clip_id": clip_id,
                "dedup_identity": identity,
                "stable_face_id": group[0].get("stable_face_id"),
                "raw_track_ids": json.dumps(sorted({action["raw_track_id"] for action in group}), ensure_ascii=False),
                "is_target_any": any(action["is_target"] for action in group),
                "rows": len(group),
                **{f"{state.lower()}_rows": counts[state] for state in STATES},
            }
        )

    return metrics, identity_rows, actions


def aggregate(per_clip: Sequence[dict]) -> dict:
    accepted = [row for row in per_clip if row.get("accepted") == "yes"]
    source = accepted or list(per_clip)
    totals = {f"{state.lower()}_rows": sum(int(row[f"{state.lower()}_rows"]) for row in source) for state in STATES}
    non_target_rows = sum(int(row["non_target_face_rows"]) for row in source)
    protected_rows = sum(int(row["protected_face_rows"]) for row in source)
    gt_total = sum(int(row.get("gt_target_boxes", 0)) for row in source)
    gt_matched = sum(int(row.get("gt_target_matched_boxes", 0)) for row in source)
    gt_missed = sum(int(row.get("gt_target_missed_boxes", 0)) for row in source)
    gt_preserve = sum(int(row.get("gt_target_preserve_boxes", 0)) for row in source)
    gt_altered = sum(int(row.get("gt_target_altered_boxes", 0)) for row in source)
    gt_swap = sum(int(row.get("gt_target_swap_boxes", 0)) for row in source)
    gt_blur = sum(int(row.get("gt_target_blur_boxes", 0)) for row in source)
    gt_other = sum(int(row.get("gt_target_other_action_boxes", 0)) for row in source)
    gt_threshold = float(source[0].get("gt_iou_threshold", 0.0)) if source else 0.0
    return {
        "clips": len(per_clip),
        "accepted_clips": len(accepted),
        "aggregate_source": "accepted" if accepted else "all",
        "protected_face_rows": protected_rows,
        "non_target_face_rows": non_target_rows,
        "states": totals,
        "gt_iou_threshold": gt_threshold,
        "gt_target_boxes": gt_total,
        "gt_target_matched_boxes": gt_matched,
        "gt_target_missed_boxes": gt_missed,
        "gt_target_preserve_boxes": gt_preserve,
        "gt_target_altered_boxes": gt_altered,
        "gt_target_swap_boxes": gt_swap,
        "gt_target_blur_boxes": gt_blur,
        "gt_target_other_action_boxes": gt_other,
        "gt_target_match_rate": round(safe_div(gt_matched, gt_total), 6) if gt_total else None,
        "gt_target_processed_as_preserve_rate": round(safe_div(gt_preserve, gt_total), 6) if gt_total else None,
        "gt_target_altered_rate": round(safe_div(gt_altered, gt_total), 6) if gt_total else None,
        "gt_target_missed_rate": round(safe_div(gt_missed, gt_total), 6) if gt_total else None,
        "protected_alteration_rate": round(
            safe_div(
                sum(int(row["protected_swap_rows"]) + int(row["protected_blur_rows"]) + int(row["protected_failed_rows"]) for row in source),
                protected_rows,
            ),
            6,
        ),
        "non_target_exposure_rate": round(
            safe_div(sum(int(row["non_target_preserve_rows"]) + int(row["non_target_unprocessed_rows"]) for row in source), non_target_rows),
            6,
        ),
        "anonymization_coverage": round(
            safe_div(sum(int(row["non_target_swap_rows"]) + int(row["non_target_blur_rows"]) for row in source), non_target_rows),
            6,
        ),
        "swap_coverage": round(safe_div(sum(int(row["non_target_swap_rows"]) for row in source), non_target_rows), 6),
        "blur_coverage": round(safe_div(sum(int(row["non_target_blur_rows"]) for row in source), non_target_rows), 6),
        "non_target_unprocessed_rate": round(safe_div(sum(int(row["non_target_unprocessed_rows"]) for row in source), non_target_rows), 6),
        "non_target_unknown_rate": round(safe_div(sum(int(row["non_target_unknown_rows"]) for row in source), non_target_rows), 6),
        "unknown_rate": round(
            safe_div(sum(int(row["unknown_rows"]) for row in source), sum(int(row["deduped_face_rows"]) for row in source)),
            6,
        ),
        "mean_target_coverage": round(statistics.mean(float(row["target_coverage"]) for row in source), 6) if source else 0.0,
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_aggregate_for_csv(payload: dict) -> dict:
    row = {key: value for key, value in payload.items() if not isinstance(value, (dict, list))}
    for state, count in payload.get("states", {}).items():
        row[state] = count
    return row


def evaluate(
    runs_dir: Path,
    output_dir: Path,
    review_csv: Path | None = None,
    *,
    blur_fallback_enabled: bool = True,
    clip_ids: Sequence[str] | None = None,
    gt_annotations: Path | None = None,
    gt_iou_threshold: float = 0.5,
) -> dict:
    run_dirs = find_run_dirs(runs_dir)
    if clip_ids is not None:
        wanted = set(clip_ids)
        run_dirs = [run_dir for run_dir in run_dirs if run_dir.name in wanted]
        if not run_dirs:
            raise EvaluationError("No selected run directories found")
    review_rows = load_review(review_csv)
    gt_rows_by_clip = load_gt_annotations(gt_annotations)
    face_paths = [path for run_dir in run_dirs if (path := find_one(run_dir, "face_metadata*.json")) is not None]
    tracking_paths = [path for run_dir in run_dirs if (path := find_one(run_dir, "tracking_metadata*.json")) is not None]

    schema = schema_report(face_paths, tracking_paths)
    per_clip: list[dict] = []
    per_identity: list[dict] = []
    per_frame_actions: list[dict] = []
    gt_match_rows: list[dict] = []
    for run_dir in run_dirs:
        clip_id = run_dir.name
        clip_metrics, identity_rows, actions = evaluate_clip(clip_id, run_dir, review_rows.get(clip_id), blur_fallback_enabled=blur_fallback_enabled)
        gt_metrics, clip_gt_matches = compute_gt_target_metrics(
            clip_id,
            actions,
            gt_rows_by_clip.get(clip_id, []),
            iou_threshold=gt_iou_threshold,
        )
        clip_metrics.update(gt_metrics)
        per_clip.append(clip_metrics)
        per_identity.extend(identity_rows)
        per_frame_actions.extend(actions)
        gt_match_rows.extend(clip_gt_matches)

    output_dir = output_dir.expanduser().resolve()
    write_csv(output_dir / "per_clip_metrics.csv", per_clip)
    write_csv(output_dir / "per_identity_metrics.csv", per_identity)
    write_csv(output_dir / "gt_target_matches.csv", gt_match_rows)
    write_jsonl(output_dir / "per_frame_actions.jsonl", per_frame_actions)
    aggregate_payload = aggregate(per_clip)
    write_json(output_dir / "aggregate_metrics.json", aggregate_payload)
    write_csv(output_dir / "aggregate_metrics.csv", [flatten_aggregate_for_csv(aggregate_payload)])
    write_json(output_dir / "schema_report.json", schema)
    config = {
        "runs_dir": str(runs_dir.expanduser().resolve()),
        "review_csv": str(review_csv.expanduser().resolve()) if review_csv else None,
        "classification_policy": {
            "final_action": "Use final_action directly when present.",
            "legacy_target": "Without final_action, is_target=True -> PRESERVE",
            "legacy_swap": "Without final_action, background row with exact frame/track log SwapSuccess=True -> SWAP",
            "legacy_blur": "Without final_action, background row with exact log SwapSuccess=False, embedding failure, non-GOOD quality, or fallback reasons -> BLUR when fallback blur is enabled",
            "legacy_unprocessed": "Without final_action, the same rows -> UNPROCESSED when fallback blur is disabled",
            "legacy_unknown": "Without final_action, background GOOD rows without per-track swap log evidence -> UNKNOWN",
        },
        "deduplication_key": "(frame_idx, stable_face_id) else (frame_idx, raw_track_id)",
        "target_coverage": "unique protected frames / total frames",
        "blur_fallback_enabled": blur_fallback_enabled,
        "selected_clip_ids": sorted(clip_ids) if clip_ids is not None else None,
        "notes": [
            "Corrected benchmark metadata contains final_action, swap_success, and blur_applied fields.",
            "For corrected metadata, action-state metrics are derived from final_action.",
            "Conservative log/quality-based inference is retained only for legacy metadata without final_action.",
        ],
    }
    write_json(output_dir / "evaluation_config.json", config)
    return {
        "output_dir": str(output_dir),
        "clips": len(per_clip),
        "aggregate_metrics": str(output_dir / "aggregate_metrics.json"),
        "aggregate_metrics_csv": str(output_dir / "aggregate_metrics.csv"),
        "gt_target_matches_csv": str(output_dir / "gt_target_matches.csv"),
        "schema_report": str(output_dir / "schema_report.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate VEIL metadata into conservative action-state metrics.")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--gt-annotations", type=Path, help="Optional GT target annotation CSV/JSON/JSONL.")
    parser.add_argument("--gt-iou-threshold", type=float, default=0.5, help="IoU threshold for matching GT target boxes to system observations.")
    parser.add_argument("--clip-id", action="append", default=None, help="Evaluate only this run directory name; may be repeated.")
    parser.add_argument(
        "--blur-fallback-enabled",
        choices=("true", "false"),
        default="true",
        help="Set false for no-blur-fallback ablations so failed/low-quality background rows count as UNPROCESSED.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        blur_fallback_enabled = args.blur_fallback_enabled == "true"
        print(json.dumps(
            evaluate(
                args.runs_dir,
                args.output_dir,
                args.review_csv,
                blur_fallback_enabled=blur_fallback_enabled,
                clip_ids=args.clip_id,
                gt_annotations=args.gt_annotations,
                gt_iou_threshold=args.gt_iou_threshold,
            ),
            ensure_ascii=False,
        ))
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
