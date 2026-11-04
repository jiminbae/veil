#!/usr/bin/env python3
"""Recover all nine rejected intervals; preserve missing reference provenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import cv2
from tools.prepare_research_inputs import ROOT,ENDPOINT,sha256,write_json


def recover(candidate,output):
    clip_id=candidate['clip_id'];folder=output/clip_id;folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'provenance.json'
    if receipt.exists():return json.loads(receipt.read_text())
    match=re.fullmatch(r'(.+)_hd_(\d{2})_(\d{2})_(\d+)',clip_id)
    sequence,panel,node,timecode=match.groups();start=int(timecode)/10000
    url=f'{ENDPOINT}/{sequence}/videos/hd_shared_crf20/hd_{panel}_{node}.mp4'
    video=folder/'input.mp4';reference=folder/'reference_frame.png';proxy=folder/'browser_review.mp4'
    row={**candidate,'sequence':sequence,'camera':f'hd_{panel}_{node}','start_seconds':start,
        'requested_duration_seconds':10.,'source_video_url':url,
        'historical_target_identity':None,'historical_target_reference_sha256':None,
        'target_reference_status':'unavailable; rejected target crop/source identity not retained',
        'processing_status':'pending_target_registration_and_reference_provenance',
        'historical_camera_glitch':candidate['reason']=='camera is blipping',
        'camera_glitch_verification':'historical review only; requires renewed human review'}
    try:
        subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n','-ss',str(start),
            '-i',url,'-t','10','-an','-c:v','mpeg4','-q:v','2','-threads','2',str(video)],check=True,timeout=300)
        subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n','-ss',str(start+5),
            '-i',url,'-frames:v','1',str(reference)],check=True,timeout=120)
        subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n','-i',str(video),
            '-an','-c:v','libx264','-preset','fast','-crf','18','-threads','2','-movflags','+faststart',str(proxy)],check=True,timeout=120)
        cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
        width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count=0
        while cap.read()[0]:count+=1
        cap.release()
        row.update(input_status='decoded' if count>=290 else 'truncated',frame_count=count,width=width,height=height,fps=fps,
            clip_path=str(video.relative_to(ROOT)),clip_sha256=sha256(video),reference_frame_sha256=sha256(reference))
    except Exception as exc:row.update(input_status='failed',failure_reason=type(exc).__name__+':'+str(exc))
    write_json(receipt,row);print(clip_id,row['input_status'],candidate['reason'],flush=True);return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    source=ROOT/'paper_results/panoptic20_ablation/source_review_base20_summary.json'
    candidates=json.loads(source.read_text())['rejected'];out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(a.workers) as pool:rows=list(pool.map(lambda r:recover(r,out),candidates))
    (out/'stress_manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    html=['<!doctype html><meta charset="utf-8"><title>Private rejected-input review</title><h1>Historical rejected intervals</h1><p>All nine are retained. Historical target identities and rejected crop bytes are unavailable. Register targets and verify camera glitches before a faithful stress evaluation.</p>']
    for row in rows:
        html.append('<h2>'+row['clip_id']+'</h2><p>'+row['reason']+'</p><video controls width="800" src="'+row['clip_id']+'/browser_review.mp4"></video><p><img width="800" src="'+row['clip_id']+'/reference_frame.png"></p>')
    (out/'stress_review.html').write_text('\n'.join(html))


if __name__=='__main__':main()
