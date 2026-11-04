#!/usr/bin/env python3
"""Exhaustive frame accounting and independent derived-label evaluation.

No output here calls projected labels human GT or system actions anonymity.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tools.evaluate_veil_metadata import bbox_iou
from tools.prepare_research_inputs import write_json,sha256


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def ratio(a,b): return a/b if b else None


def match_frame(labels,observations,threshold=.5):
    """Maximum-cardinality, then maximum-IoU matching; no duplicate reuse."""
    if not labels or not observations:return {}
    ious=np.array([[bbox_iou(g['bbox'],s.get('bbox')) for s in observations] for g in labels])
    weights=np.where(ious>=threshold,max(len(labels),len(observations))+ious,0.)
    li,oi=linear_sum_assignment(-weights)
    return {int(i):(int(j),float(ious[i,j])) for i,j in zip(li,oi) if ious[i,j]>=threshold}


def event_stats(rows,states):
    events=[]
    grouped=defaultdict(list)
    for row in rows:grouped[row['person_id']].append(row)
    for person,observations in grouped.items():
        start=last=None
        for row in sorted(observations,key=lambda r:r['frame_idx']):
            frame=row['frame_idx']
            bad=row['state'] in states
            if start is not None and (not bad or frame!=last+1):
                events.append((person,start,last));start=last=None
            if bad:
                if start is None:start=frame
                last=frame
        if start is not None:events.append((person,start,last))
    return events


def switches(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[r['person_id']].append(r)
    switched=transitions=0
    for values in grouped.values():
        values=sorted(values,key=lambda r:r['frame_idx'])
        for a,b in zip(values,values[1:]):
            if b['frame_idx']==a['frame_idx']+1 and a['state'] in {'SWAP','BLUR'} and b['state'] in {'SWAP','BLUR'}:
                transitions+=1;switched+=a['state']!=b['state']
    return switched,transitions


def make_control(row,directory):
    directory.mkdir(parents=True,exist_ok=True);path=directory/(row['clip_id']+'.mp4')
    receipt=path.with_suffix('.json')
    if receipt.exists():
        meta=json.loads(receipt.read_text())
        if meta['input_sha256']==row['clip_path_sha256'] and path.exists() and meta['sha256']==sha256(path):return path
        raise ValueError('Control receipt mismatch; use a new control directory')
    cap=cv2.VideoCapture(row['clip_path'])
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),row['fps'],(row['width'],row['height']))
    if not writer.isOpened():raise RuntimeError('Control encoder failed')
    n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        writer.write(frame);n+=1
    cap.release();writer.release()
    write_json(receipt,{'frames':n,'input_sha256':row['clip_path_sha256'],'sha256':sha256(path),
        'codec':'OpenCV mp4v default encoder; same as clean pipeline; whole-frame bitrate can couple ROIs'})
    return path


def evaluate_matrix(manifest,annotations,benchmark,output,local_output):
    population=[json.loads(s) for s in Path(manifest).read_text().splitlines() if s.strip()]
    runs=json.loads((Path(benchmark)/'run_summary.json').read_text())
    lookup={r['clip_id']:r for r in population}
    clips=[];identities=[];runtimes=[];failures=[];statuses=[];integrity=[]
    matches_dir=Path(local_output)/'matches';matches_dir.mkdir(parents=True,exist_ok=True)
    for run in runs:
        clip_id=run['clip_id'];condition=run['condition'];source=lookup[clip_id]
        statuses.append({'condition':condition,'clip_id':clip_id,'status':run['status'],
            'returncode':run['returncode'],'elapsed_sec':run['elapsed_sec']})
        if run['status'] not in {'completed','skipped_existing'}:
            failures.append({'condition':condition,'clip_id':clip_id,'failure_type':'run_'+run['status'],'count':1})
            clips.append({'condition':condition,'clip_id':clip_id,'sequence':source['sequence'],
                'status':run['status'],'expected_frames':source.get('frame_count'),'evaluated_frames':0,
                'derived_target_preserve_rate':None,'derived_nontarget_exposure_rate':None})
            continue
        run_dir=Path(run['face_metadata']).parent
        actions=json.loads(Path(run['face_metadata']).read_text())
        frames=json.loads((run_dir/'frames.json').read_text())
        runtime=json.loads((run_dir/'runtime.json').read_text())
        receipt=json.loads((run_dir/'run_receipt.json').read_text())
        annotation_path=Path(annotations)/(clip_id+'.jsonl')
        labels=[json.loads(s) for s in annotation_path.read_text().splitlines() if s.strip()] if annotation_path.exists() else []
        eligible=[g for g in labels if g['evaluation_eligible']]
        by_frame=defaultdict(list);observations=defaultdict(list)
        for g in eligible:by_frame[g['frame_idx']].append(g)
        for s in actions:observations[s['frame_idx']].append(s)
        rows=[];unmatched_system=0
        for frame_idx in range(1,source['frame_count']+1):
            gl=by_frame[frame_idx];sl=observations[frame_idx];assigned=match_frame(gl,sl)
            unmatched_system+=len(sl)-len(assigned)
            for i,g in enumerate(gl):
                j,iou=assigned.get(i,(None,None));s=sl[j] if j is not None else None
                rows.append({'condition':condition,'clip_id':clip_id,'frame_idx':frame_idx,
                    'person_id':g['person_id'],'is_target':g['is_target'],'bbox':g['bbox'],
                    'state':s.get('final_action','UNKNOWN') if s else 'MISSED',
                    'system_target':s.get('is_target_final') if s else None,
                    'raw_track_id':s.get('raw_track_id') if s else None,
                    'stable_face_id':s.get('stable_face_id') if s else None,
                    'iou':iou,'annotation_source':g['annotation_source'],
                    'verification_status':g['verification_status']})
        (matches_dir/(condition+'__'+clip_id+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        tc=Counter(r['state'] for r in rows if r['is_target'])
        nc=Counter(r['state'] for r in rows if not r['is_target'])
        sc=Counter(s['final_action'] for s in actions)
        t=sum(tc.values());n=sum(nc.values());a=sum(sc.values())
        target_events=event_stats([r for r in rows if r['is_target']],{'SWAP','BLUR'})
        exposure_events=event_stats([r for r in rows if not r['is_target']],{'PRESERVE','UNPROCESSED','MISSED','UNKNOWN'})
        switch_count,transition_count=switches([r for r in rows if not r['is_target']])
        associations=defaultdict(set)
        track_people=defaultdict(set)
        for r in rows:
            if r['stable_face_id'] is not None:
                associations[r['person_id']].add(r['stable_face_id'])
                track_people[r['stable_face_id']].add(r['person_id'])
        metrics={'condition':condition,'clip_id':clip_id,'sequence':source['sequence'],'status':'completed',
            'annotation_level':labels[0]['annotation_source'] if labels else 'not_available','human_verified':False,
            'expected_frames':source['frame_count'],'evaluated_frames':len(frames),
            'frames_without_system_observation':sum(not observations[f] for f in range(1,source['frame_count']+1)),
            'system_observations':a,'derived_target_observations':t,'derived_nontarget_observations':n,
            'system_target_frame_count':len({s['frame_idx'] for s in actions if s['is_target_final']}),
            'system_target_frame_coverage':len({s['frame_idx'] for s in actions if s['is_target_final']})/source['frame_count'],
            'system_nontarget_observations':sum(not s['is_target_final'] for s in actions),
            'annotation_ineligible_visibility_count':len(labels)-len(eligible),
            'unmatched_system_observations':unmatched_system,
            'derived_target_preserve_rate':ratio(tc['PRESERVE'],t),
            'derived_target_altered_rate':ratio(tc['SWAP']+tc['BLUR'],t),
            'derived_target_missed_rate':ratio(tc['MISSED'],t),
            'derived_nontarget_exposure_rate':ratio(nc['PRESERVE']+nc['UNPROCESSED']+nc['MISSED']+nc['UNKNOWN'],n),
            'derived_nontarget_false_preserve_rate':ratio(nc['PRESERVE'],n),
            'derived_nontarget_missed_rate':ratio(nc['MISSED'],n),
            'derived_detection_recall':ratio(t+n-tc['MISSED']-nc['MISSED'],t+n),
            'target_alteration_events':len(target_events),
            'target_alteration_longest_frames':max((b-a+1 for _,a,b in target_events),default=0),
            'target_alteration_longest_sec':max((b-a+1 for _,a,b in target_events),default=0)/source['fps'],
            'nontarget_exposure_events':len(exposure_events),
            'nontarget_exposure_longest_frames':max((b-a+1 for _,a,b in exposure_events),default=0),
            'swap_blur_switches':switch_count,'swap_blur_transitions':transition_count,
            'swap_blur_switch_rate':ratio(switch_count,transition_count),
            'stable_ids_associated_with_multiple_label_people':sum(len(v)>1 for v in track_people.values()),
            'label_people_with_multiple_stable_ids':sum(len(v)>1 for v in associations.values()),
            'swap_cache_observations':sum(bool(s.get('swap_cache_used')) for s in actions)}
        for prefix,counts in [('target',tc),('nontarget',nc),('system',sc)]:
            for state in ['PRESERVE','SWAP','BLUR','UNPROCESSED','UNKNOWN','MISSED']:
                metrics[prefix+'_'+state.lower()+'_count']=counts[state]
        for person in sorted({r['person_id'] for r in rows}):
            obs=[r for r in rows if r['person_id']==person];counts=Counter(r['state'] for r in obs)
            identity={'condition':condition,'clip_id':clip_id,'sequence':source['sequence'],
                'person_id':person,'is_target':obs[0]['is_target'],'observations':len(obs),
                'annotation_level':obs[0]['annotation_source']}
            identity.update({state.lower()+'_count':counts[state] for state in ['PRESERVE','SWAP','BLUR','UNPROCESSED','UNKNOWN','MISSED']})
            identities.append(identity)
        for label,counter in [('target',tc),('nontarget',nc)]:
            for state,count in counter.items():
                if state in ({'SWAP','BLUR','MISSED','UNKNOWN'} if label=='target' else {'PRESERVE','UNPROCESSED','MISSED','UNKNOWN'}):
                    failures.append({'condition':condition,'clip_id':clip_id,'failure_type':label+'_'+state.lower(),'count':count})
        for reason,count in Counter(reason for s in actions for reason in s.get('fallback_reasons',[])).items():
            failures.append({'condition':condition,'clip_id':clip_id,'failure_type':'system_reason_'+reason,'count':count})
        rt={'condition':condition,'clip_id':clip_id,'frames':source['frame_count'],
            'width':source['width'],'height':source['height'],'input_fps':source['fps'],
            'end_to_end_sec':run['elapsed_sec'],'end_to_end_fps':source['frame_count']/run['elapsed_sec'],
            'model_load_after_import_sec':runtime['model_load_sec'],
            'warm_wall_sec':runtime['warm_wall_sec'],'warm_fps':source['frame_count']/runtime['warm_wall_sec'],
            'startup_and_teardown_sec':run['elapsed_sec']-runtime['warm_wall_sec'],
            'peak_nvml_process_gpu_memory_bytes':receipt.get('peak_nvml_process_gpu_memory_bytes'),
            'peak_torch_gpu_memory_bytes':runtime['peak_torch_gpu_memory_bytes'],
            'actual_onnx_providers':json.dumps(runtime['recognizer_actual_providers'],sort_keys=True),
            'system_face_observations':len(actions)}
        for name in ['detect_sec','track_sec','identify_sec','swap_sec','finalize_sec','writer_enqueue_sec']:
            rt[name]=sum(f[name] for f in frames)
        runtimes.append(rt)
        integrity.append({'condition':condition,'clip_id':clip_id,
            'unique_frame_ledger':len({f['frame_idx'] for f in frames})==len(frames),
            'complete_frame_ledger':len(frames)==source['frame_count'],
            'actions_in_range':all(1<=s['frame_idx']<=source['frame_count'] for s in actions),
            'target_render_pixels_unchanged':all(s.get('render_changed_pixels')==0 for s in actions if s['is_target_final']),
            'completed_receipt':receipt['status']=='completed',
            'clean_output':runtime['clean_output'],
            'matching_denominator_reconciles':t+n==len(rows)})
        clips.append(metrics)
        make_control(source,Path(local_output)/'controls')
        print(condition,clip_id,'target',t,'nontarget',n,flush=True)
    for name,data in [('run_status.csv',statuses),('per_clip_metrics.csv',clips),
        ('per_identity_metrics.csv',identities),('runtime_metrics.csv',runtimes),('failure_cases.csv',failures)]:
        csv_write(Path(output)/name,data)
    write_json(Path(output)/'integrity_checks.json',{'checks':integrity,
        'all_run_checks_pass':all(all(v for k,v in check.items() if isinstance(v,bool)) for check in integrity),
        'completed_runs':len(integrity),'planned_population':len(population),
        'unique_condition_clip_keys':len({(r['condition'],r['clip_id']) for r in statuses})==len(statuses),
        'human_annotation_validation':False,'identity_verifier_validation':'separate scoring stage'})


def main():
    p=argparse.ArgumentParser()
    for name in ['manifest','annotations','benchmark','output-dir','local-output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();evaluate_matrix(a.manifest,a.annotations,a.benchmark,a.output_dir,a.local_output)


if __name__=='__main__':main()
