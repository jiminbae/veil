#!/usr/bin/env python3
"""Extract actual unaltered frames/ROIs privately; no human quality labels."""
import argparse
import json
from pathlib import Path
import cv2
from tools.aggregate_research_results import read_csv,number
from tools.prepare_research_inputs import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--paper-dir',type=Path,required=True);a=p.parse_args()
    run=a.run_root.resolve();paper=a.paper_dir.resolve();dest=run/'qualitative_candidates'
    if dest.exists():raise RuntimeError('Preserve previous candidates; choose a new run directory')
    dest.mkdir()
    manifest={r['clip_id']:r for r in (json.loads(s) for s in (run/'inputs/dataset_manifest.jsonl').read_text().splitlines())}
    runs=json.loads((run/'primary/run_summary.json').read_text())
    by_run={(r['condition'],r['clip_id']):r for r in runs};chosen=[]
    for backend in ['adaface','facenet']:
        rows=[]
        for path in (run/'evaluation/scores'/backend).glob('full__*/scores.json'):
            rows.extend(r for r in json.loads(path.read_text()) if not r['is_target'] and r['score'] is not None)
        if rows:
            row=max(rows,key=lambda r:r['score']);chosen.append({**row,'selection':'highest_full_nontarget_score_'+backend})
    clip_rows=[r for r in read_csv(paper/'per_clip_metrics.csv') if r['condition']=='full' and number(r.get('derived_target_preserve_rate')) is not None]
    if clip_rows:
        worst=min(clip_rows,key=lambda r:float(r['derived_target_preserve_rate']))
        matches=[json.loads(s) for s in (run/'evaluation/matches'/('full__'+worst['clip_id']+'.jsonl')).read_text().splitlines()]
        incident=next((r for r in matches if r['is_target'] and r['state'] in {'SWAP','BLUR'}),None)
        if incident:chosen.append({**incident,'selection':'lowest_projected_target_preservation_clip_first_altered_frame'})
    if not chosen:
        raise RuntimeError('No measured candidate selection is available')
    public=[]
    for i,candidate in enumerate(chosen,1):
        clip_id=candidate['clip_id'];frame_idx=candidate['frame_idx'];folder=dest/f'candidate_{i:02d}';folder.mkdir()
        labels=[json.loads(s) for s in (run/'annotations'/(clip_id+'.jsonl')).read_text().splitlines()]
        label=next(g for g in labels if g['frame_idx']==frame_idx and g['person_id']==candidate['person_id'])
        paths={'original':manifest[clip_id]['clip_path']}
        for c in ['full','no_identity_lock','no_blur_fallback','selective_blur_only','no_mouth_paste','no_swap_cache']:
            paths[c]=by_run[c,clip_id]['output_video']
        for name,path in paths.items():
            cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx-1);ok,frame=cap.read();cap.release()
            if not ok:raise RuntimeError('Cannot decode candidate frame')
            cv2.imwrite(str(folder/(name+'_frame.png')),frame)
            x1,y1,x2,y2=label['bbox'];bw,bh=x2-x1,y2-y1
            x1,y1=max(0,int(x1-.5*bw)),max(0,int(y1-.5*bh));x2,y2=min(frame.shape[1],int(x2+.5*bw)),min(frame.shape[0],int(y2+.5*bh))
            cv2.imwrite(str(folder/(name+'_roi.png')),frame[y1:y2,x1:x2])
        record={'candidate_id':f'candidate_{i:02d}','clip_id':clip_id,'frame_idx':frame_idx,
            'person_id':candidate['person_id'],'selection_rule':candidate['selection'],
            'score':candidate.get('score'),'annotation_status':'unverified_official_3d_projection',
            'human_quality_review':False,'publication_status':'not_approved; Panoptic modified imagery requires organizer permission',
            'local_directory':str(folder.relative_to(run))}
        write_json(folder/'candidate.json',record);public.append(record)
    write_json(paper/'qualitative_candidates.json',{'candidates':public,
        'images_in_git':False,'scope':'Actual decoded clean output and source frames; no aesthetic curation or human quality rating'})
    print('Extracted',len(chosen),'private candidates; no imagery copied into Git results')


if __name__=='__main__':main()
