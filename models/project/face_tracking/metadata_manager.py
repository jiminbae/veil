import json
from pathlib import Path

def make_face_data(
    frame_idx,
    raw_track_id,
    stable_face_id,
    bbox,
    smoothed_bbox,
    is_target,
    is_background,
    target_sim,
    embedding_ok,
    quality,
    fallback_reasons
):
    return {
        "frame_idx": int(frame_idx),
        "raw_track_id": int(raw_track_id),
        "stable_face_id": int(stable_face_id) if stable_face_id is not None else None,
        "bbox": [int(v) for v in bbox],
        "smoothed_bbox": [int(v) for v in smoothed_bbox],
        "is_target": bool(is_target),
        "is_background": bool(is_background),
        "target_similarity": float(target_sim),
        "embedding_ok": bool(embedding_ok),
        "quality": str(quality),
        "fallback_reasons": list(fallback_reasons)
    }


def save_metadata(metadata_path, all_face_metadata):
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_face_metadata, f, ensure_ascii=False, indent=2)