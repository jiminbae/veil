"""
Video tracking evaluation runner.

This script does not compute true accuracy metrics because there is no ground
truth annotation. It runs each model on the same video and reports comparable
runtime/log health statistics only.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2


BASE_DIR = Path(__file__).resolve().parent
EVAL_OUTPUT_DIR = BASE_DIR / "evaluation_results"
DEFAULT_VIDEO = BASE_DIR / "people_crossing.mp4"
DEFAULT_YOLO26_FACE = Path("/home/jmbae/yolo26x-face/runs/detect/runs/face/yolo26x_widerface/weights/best.pt")
DEFAULT_YOLOV8_FACE_CANDIDATES = [
    BASE_DIR / "telle/face_tracking/weights/yolov8x-face-lindevs.pt",
    BASE_DIR / "weights/yolov8x-face-lindevs.pt",
    BASE_DIR / "yolov8x-face-lindevs.pt",
]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    script: Path
    cwd: Path
    log_file: Path
    output_video: Path


@dataclass(frozen=True)
class FaceModelSpec:
    key: str
    path: Path


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    opened: bool
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class LogStats:
    log_path: Path
    frames_seen: int
    max_frame_index: int
    total_detections: int
    total_tracks: int
    unique_track_ids: int
    unique_face_ids: int
    face_assignments: int
    none_face_assignments: int
    track_lines: int
    saved_result: str | None
    saved_log: str | None

    @property
    def face_assignment_rate(self) -> float:
        return self.face_assignments / self.track_lines if self.track_lines else 0.0


@dataclass(frozen=True)
class RunResult:
    spec: ModelSpec
    face_model: FaceModelSpec
    runtime_sec: float
    returncode: int
    stats: LogStats | None
    stderr: str


MODELS: dict[str, ModelSpec] = {
    "telle": ModelSpec(
        key="telle",
        name="tracker_arcface",
        script=BASE_DIR / "telle/face_tracking/tracker_arcface.py",
        cwd=BASE_DIR / "telle/face_tracking",
        log_file=BASE_DIR / "telle/face_tracking/tracking_xface_log.txt",
        output_video=EVAL_OUTPUT_DIR / "tracker_arcface" / "tracker_arcface.mp4",
    ),
    "seojin": ModelSpec(
        key="seojin",
        name="yolo_track",
        script=BASE_DIR / "seojin/yolov8x.py",
        cwd=BASE_DIR / "seojin",
        log_file=BASE_DIR / "seojin/yolov8x.log",
        output_video=EVAL_OUTPUT_DIR / "yolo_track" / "yolo_track.mp4",
    ),
    "minhyung": ModelSpec(
        key="minhyung",
        name="person_face_arcface",
        script=BASE_DIR / "minhyung/main.py",
        cwd=BASE_DIR / "minhyung",
        log_file=BASE_DIR / "minhyung/main.log",
        output_video=EVAL_OUTPUT_DIR / "person_face_arcface" / "person_face_arcface.mp4",
    ),
}


FRAME_DET_RE = re.compile(r"Frame=(\d+)\s+Detections=(\d+)")
FRAME_TRK_RE = re.compile(r"Frame=(\d+)\s+Tracks=(\d+)")
TRACK_LINE_RE = re.compile(r"Frame=(\d+)\s+TrackID=(\d+)\s+FaceID=(None|\d+)")
SAVED_RESULT_RE = re.compile(r"Saved result to:\s*(.+)$")
SAVED_LOG_RE = re.compile(r"Saved log to:\s*(.+)$")


def read_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    opened = cap.isOpened()
    if not opened:
        return VideoInfo(video_path, False, 0, 0.0, 0, 0)

    info = VideoInfo(
        path=video_path,
        opened=True,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def variant_log_path(spec: ModelSpec, face_model: FaceModelSpec) -> Path:
    return EVAL_OUTPUT_DIR / spec.key / face_model.key / f"{spec.key}.log"


def variant_output_path(spec: ModelSpec, face_model: FaceModelSpec) -> Path:
    return EVAL_OUTPUT_DIR / spec.key / face_model.key / f"{spec.key}.mp4"


def reset_outputs(log_path: Path, output_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if log_path.exists():
        log_path.unlink()
    if output_path.exists():
        output_path.unlink()


def run_model_script(spec: ModelSpec, face_model: FaceModelSpec, video_path: Path) -> RunResult | None:
    if not spec.script.exists():
        print(f"  script not found: {spec.script}")
        return None

    log_path = variant_log_path(spec, face_model)
    output_path = variant_output_path(spec, face_model)
    reset_outputs(log_path, output_path)

    print("\n" + "=" * 80)
    print(f"  Running: {spec.name}")
    print(f"  Face   : {face_model.key} ({face_model.path})")
    print(f"  Script : {spec.script}")
    print(f"  Video  : {video_path}")
    print(f"  Log    : {log_path}")
    print("=" * 80)

    start_time = time.time()
    result = subprocess.run(
        [
            sys.executable,
            spec.script.name,
            "--video",
            str(video_path),
            "--output",
            str(output_path),
            "--log",
            str(log_path),
            "--face-model",
            str(face_model.path),
        ],
        cwd=str(spec.cwd),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start_time

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    stats = parse_tracking_log(log_path) if log_path.exists() else None
    return RunResult(
        spec=spec,
        face_model=face_model,
        runtime_sec=elapsed,
        returncode=result.returncode,
        stats=stats,
        stderr=result.stderr,
    )


def parse_tracking_log(log_path: Path) -> LogStats:
    frame_detections: dict[int, int] = {}
    frame_tracks: dict[int, int] = {}
    unique_track_ids: set[int] = set()
    unique_face_ids: set[int] = set()

    face_assignments = 0
    none_face_assignments = 0
    track_lines = 0
    saved_result: str | None = None
    saved_log: str | None = None

    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()

        match = FRAME_DET_RE.search(line)
        if match:
            frame_detections[int(match.group(1))] = int(match.group(2))
            continue

        match = FRAME_TRK_RE.search(line)
        if match:
            frame_tracks[int(match.group(1))] = int(match.group(2))
            continue

        match = TRACK_LINE_RE.search(line)
        if match:
            track_lines += 1
            unique_track_ids.add(int(match.group(2)))
            face_id = match.group(3)
            if face_id == "None":
                none_face_assignments += 1
            else:
                face_assignments += 1
                unique_face_ids.add(int(face_id))
            continue

        match = SAVED_RESULT_RE.search(line)
        if match:
            saved_result = match.group(1).strip()
            continue

        match = SAVED_LOG_RE.search(line)
        if match:
            saved_log = match.group(1).strip()
            continue

    seen_frames = set(frame_detections) | set(frame_tracks)

    return LogStats(
        log_path=log_path,
        frames_seen=len(seen_frames),
        max_frame_index=max(seen_frames, default=0),
        total_detections=sum(frame_detections.values()),
        total_tracks=sum(frame_tracks.values()),
        unique_track_ids=len(unique_track_ids),
        unique_face_ids=len(unique_face_ids),
        face_assignments=face_assignments,
        none_face_assignments=none_face_assignments,
        track_lines=track_lines,
        saved_result=saved_result,
        saved_log=saved_log,
    )


def print_report(results: list[RunResult], video_info: VideoInfo) -> None:
    print("\n" + "=" * 100)
    print("  Evaluation Report")
    print("=" * 100)

    print(
        f"  Video: {video_info.path} | {video_info.width}x{video_info.height} | "
        f"{video_info.fps:.2f} fps | metadata frames={video_info.frame_count}"
    )

    print()
    header = (
        f"  {'model':<22} {'face':<12} {'status':<7} {'sec':>8} {'frames':>9} {'miss':>6} "
        f"{'det':>8} {'trk':>8} {'trk_id':>7} {'face_id':>7} {'face%':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for result in results:
        stats = result.stats
        status = "ok" if result.returncode == 0 else f"err{result.returncode}"
        if stats is None:
            print(
                f"  {result.spec.name:<22} {result.face_model.key:<12} {status:<7} {result.runtime_sec:>8.2f} "
                f"{'-':>9} {'-':>6} {'-':>8} {'-':>8} {'-':>7} {'-':>7} {'-':>7}"
            )
            continue

        missing = max(video_info.frame_count - stats.frames_seen, 0)
        print(
            f"  {result.spec.name:<22} {result.face_model.key:<12} {status:<7} {result.runtime_sec:>8.2f} "
            f"{stats.frames_seen:>9} {missing:>6} {stats.total_detections:>8} "
            f"{stats.total_tracks:>8} {stats.unique_track_ids:>7} "
            f"{stats.unique_face_ids:>7} {stats.face_assignment_rate * 100:>6.1f}%"
        )

    print("\n  Notes")
    print("  - This report is valid for run/log consistency, not model accuracy.")
    print("  - Face variants are comparable only within the same model row family.")
    print("  - Accuracy needs ground-truth boxes/IDs for precision, recall, IDF1, MOTA, or ID switches.")
    print("  - A larger face_id count is not automatically better; it can mean identity fragmentation.")
    print("  - 'miss' is metadata frame count minus frames found in the model log.")
    print(f"\n  Output directory: {EVAL_OUTPUT_DIR}")


def resolve_model_names(names: Iterable[str] | None) -> list[str]:
    if names is None:
        return list(MODELS.keys())
    resolved = []
    for name in names:
        if name not in MODELS:
            print(f"  Unknown model: {name}")
            print(f"  Available: {', '.join(MODELS)}")
            continue
        resolved.append(name)
    return resolved


def default_face_models() -> list[FaceModelSpec]:
    face_models: list[FaceModelSpec] = []

    for path in DEFAULT_YOLOV8_FACE_CANDIDATES:
        if path.exists():
            face_models.append(FaceModelSpec("yolov8_face", path.resolve()))
            break

    if DEFAULT_YOLO26_FACE.exists():
        face_models.append(FaceModelSpec("yolo26_face", DEFAULT_YOLO26_FACE.resolve()))

    return face_models


def parse_face_model_items(items: Iterable[str] | None) -> list[FaceModelSpec]:
    if items is None:
        return default_face_models()

    face_models = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"face model must be name=path, got: {item}")
        key, raw_path = item.split("=", 1)
        key = key.strip()
        path = Path(raw_path).expanduser().resolve()
        if not key:
            raise ValueError(f"empty face model name in: {item}")
        if not path.exists():
            raise FileNotFoundError(f"face model not found for {key}: {path}")
        face_models.append(FaceModelSpec(key=key, path=path))

    return face_models


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tracking scripts and summarize log health.")
    parser.add_argument("--mode", choices=["single", "compare"], default="compare")
    parser.add_argument("--model", type=str, help="single mode model key")
    parser.add_argument("--models", nargs="+", help="compare mode model keys")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="input video used for every model")
    parser.add_argument(
        "--face-models",
        nargs="+",
        help="face detector variants as name=path. Example: yolov8=/path/yolov8-face.pt yolo26=/path/best.pt",
    )
    args = parser.parse_args()

    video_path = args.video.resolve()
    video_info = read_video_info(video_path)
    if not video_info.opened:
        print(f"Cannot open video: {video_path}")
        return 1

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        if not args.model:
            print("--model is required in single mode.")
            return 1
        model_names = resolve_model_names([args.model])
    else:
        model_names = resolve_model_names(args.models)

    try:
        face_models = parse_face_model_items(args.face_models)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1

    if not face_models:
        print("No face model weights found.")
        print("Pass them explicitly, for example:")
        print("  --face-models yolov8=/path/to/yolov8-face.pt yolo26=/home/jmbae/yolo26x-face/runs/detect/runs/face/yolo26x_widerface/weights/best.pt")
        return 1

    results: list[RunResult] = []
    for model_name in model_names:
        for face_model in face_models:
            result = run_model_script(MODELS[model_name], face_model, video_path)
            if result is not None:
                results.append(result)

    print_report(results, video_info)
    return 0 if all(result.returncode == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
