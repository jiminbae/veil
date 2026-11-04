#!/usr/bin/env python3
"""Derive explicitly unverified 2D labels from official Panoptic 3D faces."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import cv2
import numpy as np
import requests
from tools.prepare_research_inputs import ROOT,ENDPOINT,write_json,sha256

_local=threading.local()


def fetch_json(item):
    url,path=item
    if path.exists(): return json.loads(path.read_text()),None
    if not hasattr(_local,'session'): _local.session=requests.Session()
    try:
        response=_local.session.get(url,timeout=25)
        response.raise_for_status()
        payload=response.json();write_json(path,payload)
        return payload,None
    except Exception as exc: return None,type(exc).__name__+':'+str(exc)


def project_person(person,camera,width,height):
    face=person['face70']
    xyz=np.array(face['landmarks'],float).reshape(-1,3)
    score=np.array(face['averageScore'])
    errors=np.array(face['averageReproError'])
    valid=np.isfinite(xyz).all(axis=1)&(score>=.1)&(errors<=10)
    if valid.sum()<12: return None
    rotation=np.array(camera['R'],float)
    trans=np.array(camera['t'],float)
    depth=(rotation@xyz.T+trans)[2]
    valid &= depth>0
    if valid.sum()<12: return None
    uv,_=cv2.projectPoints(xyz,cv2.Rodrigues(rotation)[0],trans,
        np.array(camera['K'],float),np.array(camera['distCoef'],float))
    uv=uv.reshape(-1,2)
    low,high=uv[valid].min(0),uv[valid].max(0)
    bw,bh=high-low
    if min(bw,bh)<=1:return None
    bbox=[max(0.,low[0]-.15*bw),max(0.,low[1]-.35*bh),
          min(float(width),high[0]+.15*bw),min(float(height),high[1]+.05*bh)]
    if bbox[2]<=bbox[0] or bbox[3]<=bbox[1]:return None
    visible=sum(int(camera['node']) in cameras for cameras in face['visibility'])
    return {'person_id':str(person['id']),'bbox':[round(v,3) for v in bbox],
        'visible_landmarks':visible,'visibility':'camera_visible' if visible>=5 else 'visibility_unconfirmed',
        'evaluation_eligible':visible>=5,'confident_landmarks':int(valid.sum()),
        'mean_3d_reprojection_error':float(errors[valid].mean()),
        'annotation_source':'official_panoptic_3d_face70_projection',
        'verification_status':'unverified_projection_requires_human_review',
        'landmarks5': [uv[36:42].mean(0).tolist(),uv[42:48].mean(0).tolist(),
                      uv[30].tolist(),uv[48].tolist(),uv[54].tolist()]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(s) for s in args.manifest.read_text().splitlines() if s.strip()]
    with ThreadPoolExecutor(args.workers) as pool:
        for row in rows:
            clip_id=row['clip_id'];dest=out/(clip_id+'.jsonl')
            if row['status']!='materialized':continue
            config_path=out/'official'/row['sequence']/('calibration_'+row['sequence']+'.json')
            calib,error=fetch_json((f"{ENDPOINT}/{row['sequence']}/{config_path.name}",config_path))
            if error:
                write_json(out/(clip_id+'_status.json'),{'status':'failed','error':error});continue
            camera=next(c for c in calib['cameras'] if c['type']=='hd' and c['name']==row['camera'][3:])
            indices=list(range(row['source_first_frame_idx'],row['source_first_frame_idx']+row['frame_count']))
            ref_idx=row['source_reference_frame_idx']
            def request(idx):
                filename=f'faceRecon3D_hd{idx:08d}.json'
                return (f"{ENDPOINT}/{row['sequence']}/hdFace3d/{filename}",out/'official'/row['sequence']/filename)
            ref,err=fetch_json(request(ref_idx))
            ref_people=[project_person(v,camera,row['width'],row['height']) for v in (ref or {}).get('people',[])]
            ref_people=[v for v in ref_people if v]
            if not ref_people:
                write_json(out/(clip_id+'_status.json'),{'status':'failed','error':err or 'no_reference_projection'});continue
            selected=np.array(row['selected_reference_bbox']);center=(selected[:2]+selected[2:])/2
            nearest=min(ref_people,key=lambda v:np.linalg.norm((np.array(v['bbox'][:2])+v['bbox'][2:])/2-center))
            distance=float(np.linalg.norm((np.array(nearest['bbox'][:2])+nearest['bbox'][2:])/2-center))
            target_id=nearest['person_id']
            labels=[];failures=[]
            for frame_idx,(payload,error) in enumerate(pool.map(fetch_json,map(request,indices)),1):
                if error: failures.append({'frame_idx':frame_idx,'reason':error});continue
                for person in payload.get('people',[]):
                    label=project_person(person,camera,row['width'],row['height'])
                    if label:
                        labels.append({**label,'clip_id':clip_id,'frame_idx':frame_idx,
                            'source_frame_idx':indices[frame_idx-1],
                            'is_target':label['person_id']==target_id})
            dest.write_text(''.join(json.dumps(v,sort_keys=True)+'\n' for v in labels))
            status={'status':'derived_unverified','clip_id':clip_id,'annotation_count':len(labels),
                'requested_frames':len(indices),'fetched_frames':len(indices)-len(failures),
                'failed_frames':failures,'target_person_id':target_id,
                'target_reference_projection_center_distance_px':distance,
                'camera_calibration_sha256':sha256(config_path),'annotations_sha256':sha256(dest),
                'temporal_alignment':'source index=ceil(start_seconds*fps)+local_frame-1; human verification pending',
                'human_verified':False,'eligible_count':sum(v['evaluation_eligible'] for v in labels)}
            write_json(out/(clip_id+'_status.json'),status)
            print(clip_id,len(labels),'labels;',len(failures),'failed frames',flush=True)


if __name__=='__main__':main()
