#!/usr/bin/env python3
"""Calibrated real-model scoring with explicit failures and source baselines."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tools.independent_verifiers import FaceRecognitionVerifier,file_hash
from tools.evaluate_research_run import csv_write,ratio
from tools.prepare_research_inputs import write_json


def crop(frame,bbox):
    if frame is None:return None
    h,w=frame.shape[:2];x1,y1,x2,y2=bbox
    # Same unprocessed projected ROI for source and output; no VEIL detection.
    bw,bh=x2-x1,y2-y1
    x1,y1=max(0,int(x1-.15*bw)),max(0,int(y1-.15*bh))
    x2,y2=min(w,int(x2+.15*bw)),min(h,int(y2+.15*bh))
    if x2<=x1 or y2<=y1:return None
    return Image.fromarray(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB))


def source_embeddings(model,row,labels,cache,signature):
    cache.mkdir(parents=True,exist_ok=True)
    receipt=cache/'source_receipt.json';archive=cache/'source_embeddings.npz'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['signature']==signature and old['embeddings_sha256']==file_hash(archive):
            with np.load(archive) as data:emb={k:data[k] for k in data.files}
            return emb,old
        raise RuntimeError('Source embedding signature mismatch; choose a new local evaluation directory')
    by_frame=defaultdict(list)
    for g in labels:by_frame[g['frame_idx']].append(g)
    emb={};attempts=[]
    ref=model.extract_embedding(Image.open(row['target_image_path']).convert('RGB'))
    if ref is not None:emb['reference']=ref
    cap=cv2.VideoCapture(row['clip_path']);frame_idx=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        frame_idx+=1
        for g in by_frame.get(frame_idx,[]):
            image=crop(frame,g['bbox']);key=f"f{frame_idx}_p{g['person_id']}"
            value=model.extract_embedding(image) if image else None
            if value is not None:emb[key]=value
            attempts.append({'key':key,'frame_idx':frame_idx,'person_id':g['person_id'],
                'success':value is not None,'reason':model.last_failure if value is None else None})
    cap.release()
    np.savez_compressed(archive,**emb)
    meta={'signature':signature,'attempts':attempts,'reference_success':ref is not None,
          'embeddings_sha256':file_hash(archive)}
    write_json(receipt,meta);return emb,meta


def summarize(rows,threshold):
    successful=[r for r in rows if r['score'] is not None]
    original_ok=[r for r in rows if r['original_verified'] is True]
    verified_success=[r for r in original_ok if r['score'] is not None]
    accepted=sum(r['accepted'] is True for r in rows)
    verified_accepted=sum(r['accepted'] is True for r in original_ok)
    original_scored=[r for r in rows if r['original_baseline_score'] is not None]
    return {'attempts':len(rows),'source_embedding_successes':sum(r['source_embedding_ok'] for r in rows),
        'output_embedding_successes':sum(r['output_embedding_ok'] for r in rows),
        'score_successes':len(successful),'score_failures':len(rows)-len(successful),
        'accepted':accepted,'accept_rate_successful_pairs':ratio(accepted,len(successful)),
        'accept_rate_lower_bound_all_attempts':ratio(accepted,len(rows)),
        'accept_rate_upper_bound_all_attempts':ratio(accepted+len(rows)-len(successful),len(rows)),
        'original_baseline_scored':len(original_scored),'original_baseline_accepted':len(original_ok),
        'original_baseline_accept_rate':ratio(len(original_ok),len(original_scored)),
        'original_verified_subset_attempts':len(original_ok),
        'original_verified_subset_scored':len(verified_success),
        'original_verified_subset_accepted':verified_accepted,
        'original_verified_subset_accept_rate':ratio(verified_accepted,len(verified_success)),
        'threshold':threshold}


def score_matrix(manifest,annotations,benchmark,calibration,output,local_output,adaface_model,stride=5):
    population=[json.loads(s) for s in Path(manifest).read_text().splitlines() if s.strip()]
    runs=json.loads((Path(benchmark)/'run_summary.json').read_text())
    local=Path(local_output);local.mkdir(parents=True,exist_ok=True)
    lookup={r['clip_id']:r for r in population}
    summaries=[];identities=[];failures=[];utility=[]
    for backend in ['adaface','facenet']:
        cal_path=Path(calibration)/(backend+'_calibration.json')
        cal=json.loads(cal_path.read_text());threshold=cal['threshold']
        model=FaceRecognitionVerifier(backend,adaface_model if backend=='adaface' else None)
        if model.metadata['checkpoint_sha256']!=cal['model']['checkpoint_sha256']:
            raise RuntimeError('Verifier checkpoint differs from frozen calibration')
        write_json(Path(output)/(backend+'_model.json'),model.metadata)
        for run in runs:
            if run['status'] not in {'completed','skipped_existing'}:continue
            row=lookup[run['clip_id']];clip_id=row['clip_id'];condition=run['condition']
            ann_path=Path(annotations)/(clip_id+'.jsonl')
            labels=[json.loads(s) for s in ann_path.read_text().splitlines() if s.strip()]
            labels=[g for g in labels if g['evaluation_eligible'] and (g['frame_idx']-1)%stride==0]
            signature={'input':row['clip_path_sha256'],'reference':row['target_image_path_sha256'],
                'annotations':file_hash(ann_path),'model':model.metadata,'stride':stride,
                'adapter_code':file_hash(Path(__file__).with_name('independent_verifiers.py')),
                'scoring_code':file_hash(Path(__file__))}
            cache=local/'embeddings'/backend/clip_id
            source,source_meta=source_embeddings(model,row,labels,cache,signature)
            dest=local/'scores'/backend/(condition+'__'+clip_id)
            dest.mkdir(parents=True,exist_ok=True)
            result_signature={**signature,'calibration':file_hash(cal_path),'output':file_hash(run['output_video'])}
            receipt=dest/'score_receipt.json'
            if receipt.exists():
                old=json.loads(receipt.read_text())
                if old['signature']!=result_signature:raise RuntimeError('Score signature mismatch; preserve old scores and use a fresh evaluation directory')
                records=json.loads((dest/'scores.json').read_text())
            else:
                matches=Path(local_output)/'matches'/(condition+'__'+clip_id+'.jsonl')
                action_map={(r['frame_idx'],r['person_id']):r['state'] for r in
                    (json.loads(s) for s in matches.read_text().splitlines())}
                by_frame=defaultdict(list)
                for g in labels:by_frame[g['frame_idx']].append(g)
                enrollment={}
                for g in labels:
                    key=f"f{g['frame_idx']}_p{g['person_id']}"
                    # First successful independent source observation is a fixed
                    # enrollment; baseline excludes its identical source frame.
                    if key in source and g['person_id'] not in enrollment:
                        enrollment[g['person_id']]=(g['frame_idx'],source[key])
                cap=cv2.VideoCapture(run['output_video']);frame_idx=0;records=[];output_emb={}
                while True:
                    ok,frame=cap.read()
                    if not ok:break
                    frame_idx+=1
                    for g in by_frame.get(frame_idx,[]):
                        key=f"f{frame_idx}_p{g['person_id']}";original=source.get(key)
                        image=crop(frame,g['bbox']);processed=model.extract_embedding(image) if image else None
                        if processed is not None:output_emb[key]=processed
                        if g['is_target']:
                            attack_ref=source.get('reference');baseline_ref=attack_ref
                        else:
                            attack_ref=original
                            enrollment_frame,baseline_ref=enrollment.get(g['person_id'],(None,None))
                            if enrollment_frame==frame_idx:baseline_ref=None
                        baseline=model.similarity(baseline_ref,original) if baseline_ref is not None and original is not None else None
                        score=model.similarity(attack_ref,processed) if attack_ref is not None and processed is not None else None
                        records.append({'condition':condition,'clip_id':clip_id,'sequence':row['sequence'],
                            'backend':backend,'frame_idx':frame_idx,'person_id':g['person_id'],
                            'is_target':g['is_target'],'action':action_map.get((frame_idx,g['person_id']),'UNKNOWN'),
                            'source_embedding_ok':original is not None,'output_embedding_ok':processed is not None,
                            'score':score,'accepted':bool(score>=threshold) if score is not None else None,
                            'original_baseline_score':baseline,
                            'original_verified':bool(baseline>=threshold) if baseline is not None else None,
                            'failure_reason':None if score is not None else (
                                'source_or_reference_embedding_failed' if attack_ref is None else model.last_failure or 'output_embedding_failed')})
                cap.release()
                # Save embeddings locally for an explicit multi-frame mean attack.
                np.savez_compressed(dest/'output_embeddings.npz',**output_emb)
                write_json(dest/'scores.json',records)
                write_json(receipt,{'signature':result_signature,'attempts':len(records),'status':'completed'})
            base={'condition':condition,'clip_id':clip_id,'sequence':row['sequence'],'backend':backend,
                'annotation_level':labels[0]['annotation_source'] if labels else 'not_available','frame_stride':stride}
            for group,predicate in [('target',lambda r:r['is_target']),('non_target',lambda r:not r['is_target']),
                ('non_target_swap',lambda r:not r['is_target'] and r['action']=='SWAP'),
                ('non_target_blur',lambda r:not r['is_target'] and r['action']=='BLUR')]:
                subset=[r for r in records if predicate(r)]
                summaries.append({**base,'group':group,**summarize(subset,threshold)})
            with np.load(dest/'output_embeddings.npz') as z:out_emb={k:z[k] for k in z.files}
            for person in sorted({r['person_id'] for r in records}):
                subset=[r for r in records if r['person_id']==person];is_target=subset[0]['is_target']
                src_values=[source[f"f{r['frame_idx']}_p{person}"] for r in subset if f"f{r['frame_idx']}_p{person}" in source]
                dst_values=[out_emb[f"f{r['frame_idx']}_p{person}"] for r in subset if f"f{r['frame_idx']}_p{person}" in out_emb]
                left=source.get('reference') if is_target else (np.mean(src_values,axis=0) if src_values else None)
                right=np.mean(dst_values,axis=0) if dst_values else None
                attack_score=model.similarity(left,right) if left is not None and right is not None else None
                ident={**base,'person_id':person,'is_target':is_target,**summarize(subset,threshold),
                    'any_frame_accepted':any(r['accepted'] is True for r in subset) if any(r['score'] is not None for r in subset) else None,
                    'any_frame_risk_upper_bound':any(r['accepted'] is True or r['score'] is None for r in subset),
                    'mean_embedding_attack_score':attack_score,
                    'mean_embedding_attack_accepted':bool(attack_score>=threshold) if attack_score is not None else None,
                    'source_frames_aggregated':len(src_values),'output_frames_aggregated':len(dst_values)}
                identities.append(ident)
            for reason,count in __import__('collections').Counter(r['failure_reason'] for r in records if r['failure_reason']).items():
                failures.append({**base,'failure_type':reason,'count':count})
            print(backend,condition,clip_id,'attempts',len(records),'scores',sum(r['score'] is not None for r in records),flush=True)
        del model
    csv_write(Path(output)/'per_clip_verifier_metrics.csv',summaries)
    csv_write(Path(output)/'per_identity_verifier_metrics.csv',identities)
    csv_write(Path(output)/'verifier_failures.csv',failures)
    write_json(Path(output)/'evaluation_status.json',{'status':'actual_two_model_scoring_completed',
        'human_gt':False,'threshold_source':'separate LFW corpus empirical 1% FMR operating point',
        'frame_stride':stride,'rows':len(summaries),'runs':len(runs),'verifiers':['adaface','facenet'],
        'failure_policy':'unknown; lower/upper bounds retain all attempted faces',
        'non_target_threat_model':'same-frame source-to-output and pooled source/output mean embeddings',
        'original_baseline':'target registered reference; non-target first successful source frame vs later source frames',
        'mean_attack_limitation':'FMR threshold calibrated for single frames, not recalibrated for aggregate embeddings'})


def main():
    p=argparse.ArgumentParser()
    for name in ['manifest','annotations','benchmark','calibration','output-dir','local-output','adaface-model']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stride',type=int,default=5);a=p.parse_args()
    score_matrix(a.manifest,a.annotations,a.benchmark,a.calibration,a.output_dir,a.local_output,a.adaface_model,a.stride)


if __name__=='__main__':main()
