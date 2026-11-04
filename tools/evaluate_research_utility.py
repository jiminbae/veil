#!/usr/bin/env python3
"""Target ROI changes relative to an identical-codec unprocessed control."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from tools.evaluate_research_run import csv_write,ratio


def main():
    p=argparse.ArgumentParser()
    for name in ['manifest','annotations','benchmark','controls','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stride',type=int,default=5);a=p.parse_args()
    manifest={r['clip_id']:r for r in (json.loads(s) for s in a.manifest.read_text().splitlines())}
    runs=json.loads((a.benchmark/'run_summary.json').read_text());results=[]
    for run in runs:
        if run['status'] not in {'completed','skipped_existing'}:continue
        row=manifest[run['clip_id']]
        labels=[json.loads(s) for s in (a.annotations/(run['clip_id']+'.jsonl')).read_text().splitlines()]
        by_frame=defaultdict(list)
        for g in labels:
            if g['is_target'] and g['evaluation_eligible'] and (g['frame_idx']-1)%a.stride==0:by_frame[g['frame_idx']].append(g)
        sources=[cv2.VideoCapture(str(v)) for v in [row['clip_path'],run['output_video'],a.controls/(run['clip_id']+'.mp4')]]
        sums=np.zeros(2);channels=observations=0;frame_idx=0
        while True:
            frames=[c.read() for c in sources]
            if not all(ok for ok,_ in frames):break
            frame_idx+=1
            for g in by_frame.get(frame_idx,[]):
                x1,y1,x2,y2=map(int,g['bbox'])
                crops=[f[y1:y2,x1:x2].astype(np.float32) for _,f in frames]
                if not crops[0].size:continue
                sums[0]+=np.abs(crops[1]-crops[0]).sum()
                sums[1]+=np.abs(crops[2]-crops[0]).sum()
                channels+=crops[0].size;observations+=1
        for c in sources:c.release()
        results.append({'condition':run['condition'],'clip_id':run['clip_id'],'sequence':row['sequence'],
            'annotation_level':'unverified_projection','target_roi_observations':observations,
            'pixel_channels':channels,'processed_mae_0_255':ratio(float(sums[0]),channels),
            'control_mae_0_255':ratio(float(sums[1]),channels),
            'excess_mae_0_255':ratio(float(sums[0]-sums[1]),channels),
            'frame_stride':a.stride,'interpretation':'target pixel change; not face-swap perceptual quality'})
        print(run['condition'],run['clip_id'],observations,flush=True)
    csv_write(a.output,results)


if __name__=='__main__':main()
