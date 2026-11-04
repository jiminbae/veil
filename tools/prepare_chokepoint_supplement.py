#!/usr/bin/env python3
"""Two distinct crowded recordings; do not multiply views to pad sample size."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import tarfile
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from tools.evaluate_research_run import match_frame
from tools.prepare_research_inputs import ROOT,write_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--archives',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    out=a.output_dir.resolve();(out/'inputs').mkdir(parents=True,exist_ok=True);(out/'annotations').mkdir(exist_ok=True)
    detector=MTCNN(keep_all=True,device='cpu');manifest=[]
    for name in ['P2E_S5','P2L_S5']:
        archive=a.archives/(name+'.tar.xz');camera=name+'_C2'
        clip_id='chokepoint_'+camera;folder=out/'inputs'/clip_id
        if folder.exists():raise ValueError('Preserve previous inputs; choose a new supplement directory')
        folder.mkdir()
        with tarfile.open(archive) as outer:inner=outer.extractfile(camera+'.tar.xz').read()
        with tarfile.open(fileobj=io.BytesIO(inner)) as source:
            members={int(Path(m.name).stem):m for m in source.getmembers() if m.isfile() and m.name.endswith('.jpg')}
            def frame(idx):
                return np.asarray(Image.open(source.extractfile(members[idx])).convert('RGB'))
            candidate=None
            active=sorted(idx for idx in members if idx>=100)
            for idx in active[::10]:
                if idx+269>max(members):break
                boxes,_=detector.detect(Image.fromarray(frame(idx)))
                if boxes is not None and len(boxes)>=2:candidate=idx;break
            if candidate is None:
                write_json(folder/'input_receipt.json',{'clip_id':clip_id,'status':'ineligible','reason':'No two-face eligibility window found under fixed scan rule'})
                continue
            start=max(active[0],candidate-30);stop=start+300
            if not all(idx in members for idx in range(start,stop)):
                raise RuntimeError('Selected window crosses a real source-frame gap; do not synthesize missing frames')
            writer=cv2.VideoWriter(str(folder/'input.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(800,600))
            if not writer.isOpened():raise RuntimeError('Supplement encoder failed')
            labels=[];tracks={};next_id=1;reference=None;target_id=None;reference_box=None
            for local,idx in enumerate(range(start,stop),1):
                rgb=frame(idx);writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                boxes,prob=detector.detect(Image.fromarray(rgb))
                detected=[] if boxes is None else [{'bbox':[max(0.,float(b[0])),max(0.,float(b[1])),min(800.,float(b[2])),min(600.,float(b[3]))]} for b in boxes]
                previous=[{'bbox':v['bbox'],'person_id':k} for k,v in tracks.items() if local-v['last']<=10]
                assignment=match_frame(detected,previous,threshold=.2)
                current=[]
                for i,d in enumerate(detected):
                    if i in assignment:person=previous[assignment[i][0]]['person_id']
                    else:person=f'p{next_id:03d}';next_id+=1
                    tracks[person]={'bbox':d['bbox'],'last':local}
                    current.append({'clip_id':clip_id,'frame_idx':local,'source_frame_idx':idx,
                        'person_id':person,'bbox':d['bbox'],'visibility':'external_detector_observed',
                        'evaluation_eligible':True,'annotation_source':'independent_mtcnn_iou_tracking',
                        'verification_status':'pseudo_label_not_human_verified'})
                if idx==candidate:
                    chosen=max(current,key=lambda g:(g['bbox'][2]-g['bbox'][0])*(g['bbox'][3]-g['bbox'][1]))
                    target_id=chosen['person_id'];reference=rgb.copy();reference_box=chosen['bbox']
                labels.extend(current)
            writer.release()
            for g in labels:g['is_target']=g['person_id']==target_id
            Image.fromarray(reference).save(folder/'reference.png')
            x1,y1,x2,y2=reference_box;bw,bh=x2-x1,y2-y1
            box=[max(0,int(x1-.5*bw)),max(0,int(y1-.5*bh)),min(800,int(x2+.5*bw)),min(600,int(y2+.5*bh))]
            Image.fromarray(reference[box[1]:box[3],box[0]:box[2]]).save(folder/'target.png')
            row={'clip_id':clip_id,'dataset':'ChokePoint','sequence':name,'camera':'C2',
                'status':'materialized','width':800,'height':600,'fps':30.,'frame_count':300,
                'start_seconds':start/30,'duration_seconds':10.,'source_first_frame_idx':start,
                'source_frame_indexing':'zero-based integer filename; background-to-active gaps retained in provenance',
                'reference_timestamp_seconds':candidate/30,'target_crop_pixel_box':box,
                'selected_reference_bbox':reference_box,'target_person_id':target_id,
                'source_video_url':f'https://zenodo.org/records/815657/files/{name}.tar.xz',
                'source_archive_sha256':sha256(archive),
                'reference_selection_rule':'first frame at 10-frame scan stride after background with >=2 independently detected faces; largest face',
                'annotation_level':'MTCNN geometry-track pseudo-label; crowded sequences have no official person GT in supplied groundtruth archive',
                'reference_verification_status':'pending_human_review',
                'license':'NICTA non-commercial research license; https://arma.sourceforge.net/chokepoint/',
                'required_citation':'Wong et al., Patch-based Probabilistic Image Quality Assessment for Face Selection and Improved Video-based Face Recognition, CVPR Workshops 2011, pp.81-88',
                'derivative_notice':'VEIL research derivative: reencoded 10-second clip and model-generated pseudo-labels; not original Licensed Material'}
            for key,file in [('clip_path','input.mp4'),('target_image_path','target.png'),('reference_frame_path','reference.png')]:
                row[key]=str((folder/file).relative_to(ROOT));row[key+'_sha256']=sha256(folder/file)
            write_json(folder/'input_receipt.json',row)
            (out/'annotations'/(clip_id+'.jsonl')).write_text(''.join(json.dumps(g)+'\n' for g in labels))
            manifest.append(row);print(clip_id,'source start',start,'labels',len(labels),'tracks',next_id-1,flush=True)
    (out/'inputs/dataset_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))


if __name__=='__main__':main()
