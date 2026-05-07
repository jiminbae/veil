"""
동영상 기반 얼굴 추적 모델 평가 전용 스크립트
- 각 모델은 각자 스크립트에 그대로 둔다
- evaluate.py는 각 스크립트를 실행하고 로그/출력만 집계
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


BASE_DIR = Path(__file__).resolve().parent
EVAL_OUTPUT_DIR = BASE_DIR / "evaluation_results"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    script: Path
    cwd: Path
    log_file: Path
    video_file: Path
    output_video: Path | None = None


MODELS: dict[str, ModelSpec] = {
    "telle": ModelSpec(
        name="tracker_arcface",
        script=BASE_DIR / "telle/face_tracking/tracker_arcface.py",
        cwd=BASE_DIR / "telle/face_tracking",
        log_file=BASE_DIR / "telle/face_tracking/tracking_xface_log.txt",
        video_file=BASE_DIR / "people_crossing.mp4",
        output_video=EVAL_OUTPUT_DIR / "tracker_arcface" / "tracker_arcface.mp4",
    ),
    "seojin": ModelSpec(
        name="yolov8x",
        script=BASE_DIR / "seojin/yolov8x.py",
        cwd=BASE_DIR / "seojin",
        log_file=BASE_DIR / "seojin/yolov8x.log",
        video_file=BASE_DIR / "people_crossing.mp4",
        output_video=EVAL_OUTPUT_DIR / "yolov8x" / "yolov8x.mp4",
    ),
}


FRAME_DET_RE = re.compile(r"Frame=(\d+)\s+Detections=(\d+)")
FRAME_TRK_RE = re.compile(r"Frame=(\d+)\s+Tracks=(\d+)")
TRACK_LINE_RE = re.compile(r"Frame=(\d+)\s+TrackID=(\d+)\s+FaceID=(\d+)")
SAVED_RESULT_RE = re.compile(r"Saved result to:\s*(.+)$")
SAVED_LOG_RE = re.compile(r"Saved log to:\s*(.+)$")


def run_model_script(spec: ModelSpec) -> dict[str, object] | None:
    if not spec.script.exists():
        print(f"  스크립트를 찾을 수 없습니다: {spec.script}")
        return None

    print(f"\n{'=' * 80}")
    print(f"  평가 실행: {spec.name}")
    print(f"  스크립트: {spec.script}")
    print(f"  작업 디렉터리: {spec.cwd}")
    print(f"{'=' * 80}")

    output_video = spec.output_video or (EVAL_OUTPUT_DIR / spec.name / f"{spec.name}.mp4")
    output_video.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    result = subprocess.run(
        [
            sys.executable,
            spec.script.name,
            "--video",
            str(spec.video_file),
            "--output",
            str(output_video),
            "--log",
            str(spec.log_file),
        ],
        cwd=str(spec.cwd),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start_time

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print(f"  실행 실패: return code={result.returncode}")
        return {
            "name": spec.name,
            "runtime_sec": elapsed,
            "returncode": result.returncode,
            "log_stats": None,
        }

    log_path = spec.log_file
    if not log_path.exists():
        print(f"  로그 파일 없음: {log_path}")
        log_stats = None
    else:
        log_stats = parse_tracking_log(log_path)

    summary = {
        "name": spec.name,
        "runtime_sec": elapsed,
        "returncode": result.returncode,
        "log_stats": log_stats,
    }
    return summary


def parse_tracking_log(log_path: Path) -> dict[str, object]:
    frame_detections: dict[int, int] = {}
    frame_tracks: dict[int, int] = {}
    unique_track_ids: set[int] = set()
    unique_face_ids: set[int] = set()

    saved_result: str | None = None
    saved_log: str | None = None

    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()

        match = FRAME_DET_RE.search(line)
        if match:
            frame = int(match.group(1))
            detections = int(match.group(2))
            frame_detections[frame] = detections
            continue

        match = FRAME_TRK_RE.search(line)
        if match:
            frame = int(match.group(1))
            tracks = int(match.group(2))
            frame_tracks[frame] = tracks
            continue

        match = TRACK_LINE_RE.search(line)
        if match:
            unique_track_ids.add(int(match.group(2)))
            unique_face_ids.add(int(match.group(3)))
            continue

        match = SAVED_RESULT_RE.search(line)
        if match:
            saved_result = match.group(1).strip()
            continue

        match = SAVED_LOG_RE.search(line)
        if match:
            saved_log = match.group(1).strip()
            continue

    frame_count = max(frame_detections.keys() | frame_tracks.keys(), default=0)
    total_detections = sum(frame_detections.values())
    total_tracks = sum(frame_tracks.values())

    return {
        "log_path": str(log_path),
        "frames": frame_count,
        "total_detections": total_detections,
        "total_tracks": total_tracks,
        "avg_detections_per_frame": total_detections / frame_count if frame_count else 0.0,
        "avg_tracks_per_frame": total_tracks / frame_count if frame_count else 0.0,
        "unique_track_ids": len(unique_track_ids),
        "unique_face_ids": len(unique_face_ids),
        "saved_result": saved_result,
        "saved_log": saved_log,
    }


def compare_models(model_names: Iterable[str] | None = None) -> list[dict[str, object]]:
    if model_names is None:
        model_names = MODELS.keys()

    results: list[dict[str, object]] = []

    for model_name in model_names:
        spec = MODELS.get(model_name)
        if spec is None:
            print(f"  등록되지 않은 모델: {model_name}")
            continue

        result = run_model_script(spec)
        if result is not None:
            results.append(result)

    print("\n" + "=" * 80)
    print("  평가 결과 비교")
    print("=" * 80)

    if not results:
        print("  비교할 결과가 없음")
        return results

    header = (
        f"  {'모델명':<24} {'실행 시간(s)':>12} {'프레임 수':>8} {'총 감지 수':>10} "
        f"{'총 트랙 수':>10} {'고유 트랙 ID':>10} {'고유 Face ID':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    best_runtime = min(results, key=lambda item: item["runtime_sec"])
    best_unique_faces = max(
        (item for item in results if item.get("log_stats")),
        key=lambda item: item["log_stats"]["unique_face_ids"],
        default=None,
    )

    for item in results:
        stats = item.get("log_stats")
        if not stats:
            print(f"  {item['name']:<24} {item['runtime_sec']:>12.2f} {'-':>8} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
            continue

        print(
            f"  {item['name']:<24} {item['runtime_sec']:>12.2f} {stats['frames']:>8} "
            f"{stats['total_detections']:>10} {stats['total_tracks']:>10} "
            f"{stats['unique_track_ids']:>10} {stats['unique_face_ids']:>10}"
        )

    print("\n  🏆 최단 실행 시간:")
    print(f"     - {best_runtime['name']} ({best_runtime['runtime_sec']:.2f}s)")

    if best_unique_faces is not None:
        print("  🏆 가장 많은 고유 Face ID:")
        print(
            f"     - {best_unique_faces['name']} "
            f"({best_unique_faces['log_stats']['unique_face_ids']}개)"
        )

    print("\n  지표 정의:")
    print("     - 실행 시간(s): 스크립트 전체 실행 시간")
    print("     - 프레임 수: 로그에 기록된 처리 프레임 수")
    print("     - 총 감지 수: 모든 프레임의 detection 합")
    print("     - 총 트랙 수: 모든 프레임의 track 합")
    print("     - 고유 트랙 ID: 서로 다른 track id 개수")
    print("     - 고유 Face ID: 서로 다른 stable face id 개수")

    print(f"\n  결과 디렉터리: {EVAL_OUTPUT_DIR}")
    return results


def print_single_result(result: dict[str, object]) -> None:
    stats = result.get("log_stats")

    print("\n" + "=" * 80)
    print("  📈 평가 결과")
    print("=" * 80)

    header = (
        f"  {'모델명':<24} {'실행 시간(s)':>12} {'프레임 수':>8} {'총 감지 수':>10} "
        f"{'총 트랙 수':>10} {'고유 트랙 ID':>10} {'고유 Face ID':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    if not stats:
        print(f"  {result['name']:<24} {result['runtime_sec']:>12.2f} {'-':>8} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
        print(f"\n  결과 디렉터리: {EVAL_OUTPUT_DIR}")
        return

    print(
        f"  {result['name']:<24} {result['runtime_sec']:>12.2f} {stats['frames']:>8} "
        f"{stats['total_detections']:>10} {stats['total_tracks']:>10} "
        f"{stats['unique_track_ids']:>10} {stats['unique_face_ids']:>10}"
    )

    print("\n  지표 정의:")
    print("     - 실행 시간(s): 스크립트 전체 실행 시간")
    print("     - 프레임 수: 로그에 기록된 처리 프레임 수")
    print("     - 총 감지 수: 모든 프레임의 detection 합")
    print("     - 총 트랙 수: 모든 프레임의 track 합")
    print("     - 고유 트랙 ID: 서로 다른 track id 개수")
    print("     - 고유FaceID: 서로 다른 stable face id 개수")
    print(f"\n  💾 결과 디렉터리: {EVAL_OUTPUT_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description="동영상 기반 얼굴 추적 모델 평가 전용 스크립트")
    parser.add_argument("--mode", choices=["single", "compare"], default="compare")
    parser.add_argument("--model", type=str, help="single 모드에서 평가할 모델명")
    parser.add_argument("--models", nargs="+", help="compare 모드에서 평가할 모델명 목록")
    args = parser.parse_args()

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        if not args.model:
            print("--model 옵션 필요.")
            return 1
        if args.model not in MODELS:
            print(f" 등록되지 않은 모델: {args.model}")
            print(f"   사용 가능: {', '.join(MODELS.keys())}")
            return 1

        result = run_model_script(MODELS[args.model])
        if result is not None:
            print_single_result(result)
        return 0 if result is not None and result.get("returncode") == 0 else 1

    compare_models(args.models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
