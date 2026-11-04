#!/usr/bin/env python3
"""Reconstruct the fixed Panoptic population without publishing restricted media."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import cv2
import requests

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = 'http://domedb.perception.cs.cmu.edu/webdata/dataset'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def population():
    historical = ROOT / 'paper_results/panoptic20_ablation/accepted_manifest.jsonl'
    rows = [json.loads(s) for s in historical.read_text().splitlines() if s.strip()]
    summary = json.loads((historical.parent / 'accepted_summary.json').read_text())
    extra = summary['excluded_from_accepted_21']
    neighbor = next(r for r in rows if r['sequence'] == extra['sequence']
                    and r['start_seconds'] == 125)
    rows.append({**neighbor, **extra, 'duration_seconds': 10.0,
        'reference_timestamp_seconds': 140.0, 'target_manually_replaced': False,
        'reinstated': True,
        'reference_selection_rule': 'nearest independent face to previous interval reference center'})
    return sorted(rows, key=lambda r: r['clip_id'])


def materialize(row, output):
    row = dict(row)
    clip_dir = output / row['clip_id']
    clip_dir.mkdir(parents=True, exist_ok=True)
    receipt = clip_dir / 'input_receipt.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old.get('status') == 'materialized' and all(
            Path(old[k]).exists() and sha256(old[k]) == old[k + '_sha256']
            for k in ('clip_path', 'target_image_path', 'reference_frame_path')):
            return old
        raise RuntimeError(f'Input receipt failed; preserve attempt and choose a new output directory: {clip_dir}')
    started = time.monotonic()
    video = clip_dir / 'input.mp4'
    reference = clip_dir / 'reference.png'
    target = clip_dir / 'target.png'
    url = f"{ENDPOINT}/{row['sequence']}/videos/hd_shared_crf20/{row['camera']}.mp4"
    row.update(source_video_url=url, status='failed', historical_clip=not row.get('reinstated', False))
    try:
        head = requests.head(url, timeout=30)
        head.raise_for_status()
        row['source_http_headers'] = {k: head.headers.get(k) for k in ('Content-Length','ETag','Last-Modified')}
        # Input -ss uses range requests; exact seeking decodes from the preceding keyframe.
        cmd = ['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n',
               '-ss',str(row['start_seconds']),'-i',url,'-t',str(row['duration_seconds']),
               '-an','-c:v','mpeg4','-q:v','2','-threads','2',str(video)]
        subprocess.run(cmd, check=True, timeout=300)
        subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n',
            '-ss',str(row['reference_timestamp_seconds']),'-i',url,'-frames:v','1',str(reference)],
            check=True, timeout=120)
        image = cv2.imread(str(reference))
        if row.get('reinstated'):
            from facenet_pytorch import MTCNN
            from PIL import Image
            import numpy as np
            boxes, _ = MTCNN(keep_all=True, device='cpu').detect(Image.fromarray(image[:,:,::-1]))
            if boxes is None:
                raise RuntimeError('Independent reference detector found no face for reinstated interval')
            old_box = row['reference_faces'][row['target_face_index']]['bbox']
            center = (np.array(old_box[:2]) + old_box[2:]) / 2
            box = min(boxes, key=lambda b: np.linalg.norm((b[:2]+b[2:])/2-center))
            bw, bh = box[2]-box[0], box[3]-box[1]
            row['selected_reference_bbox'] = box.tolist()
            row['target_crop_pixel_box'] = [int(box[0]-.5*bw), int(box[1]-.5*bh),
                                            int(box[2]+.5*bw), int(box[3]+.5*bh)]
        else:
            row['selected_reference_bbox'] = row['reference_faces'][row['target_face_index']]['bbox']
        x1,y1,x2,y2 = row['target_crop_pixel_box']
        h,w = image.shape[:2]
        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
        if not cv2.imwrite(str(target),image[y1:y2,x1:x2]):
            raise RuntimeError('Could not write reference crop')
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = 0
        while cap.read()[0]: count += 1
        cap.release()
        if count < 290: raise RuntimeError(f'Truncated ten-second clip: {count} frames')
        row.update(status='materialized',frame_count=count,width=w,height=h,fps=fps,
            source_first_frame_idx=math.ceil(row['start_seconds']*fps-1e-6),
            source_reference_frame_idx=math.ceil(row['reference_timestamp_seconds']*fps-1e-6),
            reference_verification_status='pending_human_review',
            historical_reference_identity='unverifiable_no_original_image_or_hash',
            source_full_video_sha256=None, source_full_video_hash_reason='HTTP ranged extraction; full original not downloaded')
        for key,path in [('clip_path',video),('target_image_path',target),('reference_frame_path',reference)]:
            row[key] = str(path.relative_to(ROOT))
            row[key+'_sha256'] = sha256(path)
    except Exception as exc:
        row['failure_reason'] = str(exc)
    row['materialize_sec'] = time.monotonic()-started
    write_json(receipt,row)
    print(row['clip_id'],row['status'],row.get('failure_reason',''),flush=True)
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    out=args.output_dir.resolve()
    with ThreadPoolExecutor(args.workers) as pool:
        rows=list(pool.map(lambda row: materialize(row,out),population()))
    path=out/'dataset_manifest.jsonl'
    path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    print(json.dumps({'total':len(rows),'materialized':sum(r['status']=='materialized' for r in rows),'manifest':str(path)}))


if __name__=='__main__': main()
