#!/usr/bin/env python3
"""Package only actual lightweight evidence and independently check raw ledgers."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import datetime
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import numpy as np
from tools.prepare_research_inputs import ROOT,write_json,sha256
from tools.evaluate_research_run import csv_write
from tools.aggregate_research_results import read_csv,number
from tools.run_veil_benchmark import CONDITIONS


def redact(value):
    if isinstance(value,str):return value.replace(str(Path.home()),'$HOME').replace(str(ROOT),'$REPOSITORY')
    if isinstance(value,list):return [redact(v) for v in value]
    if isinstance(value,dict):return {k:redact(v) for k,v in value.items()}
    return value


def manifest_public(rows):
    keys=['clip_id','dataset','sequence','camera','source_video_url','start_seconds','duration_seconds',
        'frame_count','width','height','fps','source_first_frame_idx','source_reference_frame_idx',
        'source_full_video_sha256','source_full_video_hash_reason','source_http_headers','source_archive_sha256',
        'clip_path','clip_path_sha256','reference_frame_path','reference_frame_path_sha256',
        'reference_timestamp_seconds','target_image_path','target_image_path_sha256',
        'target_crop_pixel_box','selected_reference_bbox','target_face_index','target_manually_replaced',
        'reinstated','reference_selection_rule','reference_verification_status',
        'historical_reference_identity','status','historical_clip','annotation_level','license','derivative_notice']
    return [{**{k:r.get(k) for k in keys if k in r},'actual_duration_seconds':r['frame_count']/r['fps'],
        'source_last_frame_idx_inclusive':r['source_first_frame_idx']+r['frame_count']-1} for r in rows]


def validate_raw(base,paper,benchmark,manifest):
    runs=json.loads((benchmark/'run_summary.json').read_text());clips=read_csv(paper/'per_clip_metrics.csv')
    lookup={(r['condition'],r['clip_id']):r for r in clips};issues=[];checks=[];provenance=[]
    all_code=[];all_models=[]
    for run in runs:
        key=(run['condition'],run['clip_id'])
        if key not in lookup:issues.append(f'missing clip result: {key}');continue
        if run['status'] not in {'completed','skipped_existing'}:continue
        directory=Path(run['face_metadata']).parent
        receipt=json.loads((directory/'run_receipt.json').read_text())
        actions=json.loads(Path(run['face_metadata']).read_text())
        counts=Counter(r['final_action'] for r in actions);metric=lookup[key]
        if int(metric['system_observations'])!=len(actions):issues.append(f'system denominator mismatch: {key}')
        for state in ['PRESERVE','SWAP','BLUR','UNPROCESSED','UNKNOWN']:
            if int(metric['system_'+state.lower()+'_count'])!=counts[state]:issues.append(f'system state mismatch: {key} {state}')
        for name,digest in receipt['outputs'].items():
            if sha256(directory/name)!=digest:issues.append(f'output hash mismatch: {key} {name}')
        raw_matches=base/'evaluation/matches'/(run['condition']+'__'+run['clip_id']+'.jsonl')
        matches=[json.loads(s) for s in raw_matches.read_text().splitlines()]
        system_keys=[(r['frame_idx'],r['raw_track_id']) for r in matches if r['raw_track_id'] is not None]
        if len(system_keys)!=len(set(system_keys)):issues.append(f'one-to-one matching violated: {key}')
        for target,prefix in [(True,'target'),(False,'nontarget')]:
            subset=[r for r in matches if r['is_target']==target];states=Counter(r['state'] for r in subset)
            if int(metric['derived_'+prefix+'_observations'])!=len(subset):issues.append(f'label denominator mismatch: {key} {prefix}')
            for state in ['PRESERVE','SWAP','BLUR','UNPROCESSED','UNKNOWN','MISSED']:
                if int(metric[prefix+'_'+state.lower()+'_count'])!=states[state]:issues.append(f'label state mismatch: {key} {prefix} {state}')
        all_code.append(receipt['fingerprint']['code']);all_models.append(receipt['fingerprint']['models'])
        provenance.append({'condition':run['condition'],'clip_id':run['clip_id'],
            'processing_input_sha256':receipt['fingerprint']['clip'],
            'protected_reference_sha256':receipt['fingerprint']['protected_reference'],
            'replacement_sha256':receipt['fingerprint']['replacement'],
            'code_hashes':receipt['fingerprint']['code'],'model_hashes':receipt['fingerprint']['models'],
            'output_hashes':receipt['outputs'],'subprocess_elapsed_sec':receipt['elapsed_sec']})
    if all_code and any(v!=all_code[0] for v in all_code):issues.append('processing source hashes differ between conditions/clips')
    if all_models and any(v!=all_models[0] for v in all_models):issues.append('processing model hashes differ between conditions/clips')
    for row in manifest:
        for key in ['clip_path','target_image_path','reference_frame_path']:
            if sha256(ROOT/row[key])!=row[key+'_sha256']:issues.append(f'input hash mismatch: {row["clip_id"]} {key}')
        ann=base/'annotations'/(row['clip_id']+'.jsonl')
        labels=[json.loads(s) for s in ann.read_text().splitlines()]
        ids=[(r['frame_idx'],r['person_id']) for r in labels]
        if len(ids)!=len(set(ids)):issues.append(f'duplicate annotation face key: {row["clip_id"]}')
    verifier=read_csv(paper/'independent_identity_verifier/per_clip_verifier_metrics.csv')
    for summary in verifier:
        records=json.loads((base/'evaluation/scores'/summary['backend']/(summary['condition']+'__'+summary['clip_id'])/'scores.json').read_text())
        group=summary['group']
        subset=[r for r in records if (r['is_target'] if group=='target' else
            not r['is_target'] and (group=='non_target' or r['action']==group.removeprefix('non_target_').upper()))]
        cal=json.loads((base.parent/'calibration/results'/(summary['backend']+'_calibration.json')).read_text()) if base.name=='supplement' else json.loads((base/'calibration/results'/(summary['backend']+'_calibration.json')).read_text())
        threshold=cal['threshold'];valid=[r for r in subset if r['score'] is not None]
        accepted=sum(r['score']>=threshold for r in valid)
        if len(subset)!=int(summary['attempts']) or len(valid)!=int(summary['score_successes']) or accepted!=int(summary['accepted']):
            issues.append(f'raw verifier recomputation mismatch: {summary["backend"]} {summary["condition"]} {summary["clip_id"]} {group}')
        for field,expected in [('source_embedding_successes',sum(r['source_embedding_ok'] for r in subset)),
            ('output_embedding_successes',sum(r['output_embedding_ok'] for r in subset)),
            ('score_failures',len(subset)-len(valid))]:
            if int(summary[field])!=expected:issues.append(f'verifier extraction denominator mismatch: {summary["clip_id"]} {field}')
        if any(r['accepted'] is not None for r in subset if r['score'] is None):issues.append('failed extraction counted as a scored decision')
    if (paper/'aggregate_metrics.csv').exists():
        aggregates=read_csv(paper/'aggregate_metrics.csv')
        keys=[(r['condition'],r['metric'],r['aggregation'],r.get('backend',''),r.get('group','')) for r in aggregates]
        if len(keys)!=len(set(keys)):issues.append('duplicate aggregate keys')
        # Independently express consequential formulas rather than importing the
        # aggregation implementation or reusing its metric-definition mapping.
        formulas={
            'derived_target_preserve_rate':lambda r:(int(r['target_preserve_count']),int(r['derived_target_observations'])),
            'derived_nontarget_exposure_rate':lambda r:(sum(int(r['nontarget_'+s+'_count']) for s in ['preserve','unprocessed','missed','unknown']),int(r['derived_nontarget_observations'])),
            'system_target_frame_coverage':lambda r:(int(r['system_target_frame_count']),int(r['expected_frames'])),
            'system_nontarget_swap_fraction':lambda r:(int(r['system_swap_count']),int(r['system_nontarget_observations'])),
            'system_nontarget_blur_fraction':lambda r:(int(r['system_blur_count']),int(r['system_nontarget_observations']))}
        for result in aggregates:
            expected=None;checked=False
            if result['metric'] in formulas and not result.get('backend'):
                observations=[r for r in clips if r['condition']==result['condition'] and r['status']=='completed']
                values=[formulas[result['metric']](r) for r in observations]
                if result['aggregation']=='micro':
                    numerator=sum(n for n,d in values);denominator=sum(d for n,d in values)
                    expected=numerator/denominator if denominator else None;checked=True
                elif result['aggregation']=='clip_macro':
                    ratios=[n/d for n,d in values if d]
                    expected=sum(ratios)/len(ratios) if ratios else None;checked=True
            elif result['metric']=='accept_rate_successful_pairs' and result['aggregation']=='micro':
                group=[r for r in verifier if r['condition']==result['condition'] and r['backend']==result['backend'] and r['group']==result['group']]
                denominator=sum(int(r['score_successes']) for r in group)
                expected=sum(int(r['accepted']) for r in group)/denominator if denominator else None;checked=True
            actual=number(result['value'])
            if checked and ((expected is None)!=(actual is None) or expected is not None and abs(expected-actual)>1e-12):
                issues.append(f'aggregate independent recomputation mismatch: {result["condition"]} {result["metric"]} {result["aggregation"]}')
    csv_write(paper/'validation_checks.csv',[{'check':'raw_state_counts_and_denominators','status':'passed' if not issues else 'failed','detail_count':len(issues)},
        {'check':'matched_observation_unique_per_frame','status':'passed' if not any('one-to-one' in e for e in issues) else 'failed'},
        {'check':'model_and_processing_code_consistent','status':'passed' if not any('hashes differ' in e for e in issues) else 'failed'},
        {'check':'real_verifier_raw_score_recomputation','status':'passed' if not any('verifier' in e for e in issues) else 'failed'}])
    (paper/'per_run_provenance.jsonl').write_text(''.join(json.dumps(p,sort_keys=True)+'\n' for p in provenance))
    integrity=json.loads((paper/'integrity_checks.json').read_text());integrity.update(
        independent_raw_recomputation_passed=not issues,raw_recomputation_issues=issues,
        output_hashes_rechecked=True,input_hashes_rechecked=True,
        processing_code_hashes_consistent=not any('source hashes differ' in e for e in issues),
        processing_model_hashes_consistent=not any('model hashes differ' in e for e in issues))
    # Keep the controller's completed-stage artifact immutable; the independent
    # validation is a separate receipt, so a later resume can verify both.
    write_json(paper/'raw_validation.json',integrity)
    return issues,all_models[0] if all_models else {}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--paper-dir',type=Path,required=True);a=p.parse_args()
    run=a.run_root.resolve();paper=a.paper_dir.resolve()
    status=json.loads((run/'postprocessing_status.json').read_text())
    if status['status']!='completed':raise RuntimeError('Do not package an unfinished controller as complete')
    manifest=[json.loads(s) for s in (run/'inputs/dataset_manifest.jsonl').read_text().splitlines()]
    public=manifest_public(manifest)
    (paper/'dataset_manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in public))
    references=[];annotation_status=[];overlaps=[]
    for row in manifest:
        references.append({'clip_id':row['clip_id'],'historical_manual_replacement':row.get('target_manually_replaced',False),
            'reference_timestamp_seconds':row['reference_timestamp_seconds'],'crop_box':json.dumps(row['target_crop_pixel_box']),
            'new_reference_sha256':row['target_image_path_sha256'],'historical_reference_bytes_available':False,
            'historical_reference_reproduction_verified':False,'new_reference_verification_status':'pending_human_review'})
        metadata=json.loads((run/'annotations'/(row['clip_id']+'_status.json')).read_text())
        annotation_status.append({k:v for k,v in metadata.items() if k!='failed_frames'}|{'failed_frame_count':len(metadata['failed_frames'])})
    for i,x in enumerate(public):
        for y in public[i+1:]:
            if x['sequence']==y['sequence'] and x['camera']==y['camera']:
                count=min(x['source_last_frame_idx_inclusive'],y['source_last_frame_idx_inclusive'])-max(x['source_first_frame_idx'],y['source_first_frame_idx'])+1
                if count>0:overlaps.append({'clip_a':x['clip_id'],'clip_b':y['clip_id'],'shared_source_frames':count})
    csv_write(paper/'reference_reconstruction_audit.csv',references)
    csv_write(paper/'annotation_provenance.csv',annotation_status)
    write_json(paper/'dataset_integrity.json',{'clips':len(public),'unique_clip_ids':len({r['clip_id'] for r in public}),
        'unique_clip_hashes':len({r['clip_path_sha256'] for r in public}),'boundary_frame_overlaps':overlaps,
        'source_intervals':'nominal ten seconds; actual frame timestamps/duration disclosed',
        'human_validated_annotations':0,'requested_official_annotation_frames':sum(r['requested_frames'] for r in annotation_status),
        'fetched_official_annotation_frames':sum(r['fetched_frames'] for r in annotation_status)})
    spec=importlib.util.spec_from_file_location('snapshot_config',ROOT/'models/veil/config.py')
    config=importlib.util.module_from_spec(spec);spec.loader.exec_module(config)
    common={k:v for k,v in vars(config).items() if k.isupper() and not any(s in k for s in ['PATH','DIR','PATTERN','RUN_INDEX']) and isinstance(v,(str,int,float,bool,list,tuple,type(None)))}
    write_json(paper/'condition_manifest.json',{'conditions':{name:{**values,'changes_from_full':{k:v for k,v in values.items() if k!='description' and v!=CONDITIONS['full'].get(k)}} for name,values in CONDITIONS.items()},
        'shared_config':common,'no_identity_lock_note':'embedding cache, tracking and stable-ID association remain; not frame-independent',
        'no_swap_cache_status':'applicable: cache is read by render path; actual uses appear in per_clip_metrics',
        'clean_output':True,'workers':1,'per_clip_paths':'input manifest; target registration and replacement are distinct'})
    caldir=paper/'verifier_calibration';caldir.mkdir(exist_ok=True)
    distributions=[]
    for backend in ['adaface','facenet']:
        shutil.copy2(run/'calibration/results'/(backend+'_calibration.json'),caldir/(backend+'_calibration.json'))
        scores=read_csv(run/'calibration/results'/(backend+'_pair_scores.csv'))
        for same,label in [('True','genuine'),('False','impostor')]:
            values=[float(r['score']) for r in scores if r['same_identity']==same]
            bins=np.linspace(-1,1,41);hist,_=np.histogram(values,bins)
            distributions.extend({'backend':backend,'pair_type':label,'lower_cosine':float(bins[i]),'upper_cosine':float(bins[i+1]),'pairs':int(n)} for i,n in enumerate(hist))
    csv_write(caldir/'score_distributions.csv',distributions)
    shutil.copy2(run/'primary/benchmark_summary.json',paper/'benchmark_summary.json')
    summary=json.loads((paper/'benchmark_summary.json').read_text())
    write_json(paper/'benchmark_summary.json',redact(summary))
    issues,models=validate_raw(run,paper,run/'primary',manifest)
    if (run/'supplement/benchmark/benchmark_summary.json').exists():
        base=run/'supplement';dest=paper/'chokepoint_supplement'
        secondary=[json.loads(s) for s in (base/'inputs/dataset_manifest.jsonl').read_text().splitlines()]
        (dest/'dataset_manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in manifest_public(secondary)))
        extra,_=validate_raw(base,dest,base/'benchmark',secondary);issues.extend(extra)
        write_json(dest/'benchmark_summary.json',redact(json.loads((base/'benchmark/benchmark_summary.json').read_text())))
        (dest/'README.md').write_text('# ChokePoint supplement\n\nTwo distinct crowded recordings, one camera per recording, 300 frames each, six conditions (12 runs). Inputs are NICTA research derivatives; retain attribution and license notices. Labels are independent MTCNN/geometry-track pseudo-labels with fragmented IDs, not official or human GT. The verifier shares MTCNN detection with this label source. Treat target/identity and person-clip metrics as exploratory. Do not merge with Panoptic or infer 10–20 independent scenes.\n\nPer-clip action/runtime/utility and two-model verification summaries are in this directory. Groundtruth crowded-sequence labels were absent from the official supplied archive. Human identity review is pending.\n')
    if (run/'stress/stress_manifest.jsonl').exists():
        stress=[json.loads(s) for s in (run/'stress/stress_manifest.jsonl').read_text().splitlines()]
        (paper/'historical_stress_manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in stress))
    processing=json.loads((run/'primary_job.json').read_text());evaluation=json.loads((run/'postprocessing_job.json').read_text())
    registry={'processing_model_sha256':models,'processing_sources':{
        'yolo26x-face.pt':'https://github.com/jiminbae/yolo26x-face/releases/tag/v0.1.0',
        'inswapper_128.onnx':'https://huggingface.co/ezioruan/inswapper_128.onnx',
        'buffalo_l':'https://github.com/deepinsight/insightface/tree/master/model_zoo'},
        'independent_models':{b:json.loads((paper/'independent_identity_verifier'/(b+'_model.json')).read_text()) for b in ['adaface','facenet']},
        'license_notes':{'InsightFace_pretrained':'non-commercial research weights; no redistribution here',
            'AdaFace':'official code MIT; checkpoint/training-data terms separate; research use; no weights redistributed',
            'FaceNet':'facenet-pytorch code MIT; VGGFace2 training-data rights separate; no weights redistributed',
            'YOLO':'project author release; Ultralytics code terms and training-data rights are separate'},
        'unused_processing_fingerprint_file':'AdaFace checkpoint is hashed alongside local weights but used only by the independent verifier, not by VEIL processing'}
    facenet=importlib.util.find_spec('facenet_pytorch')
    folder=Path(next(iter(facenet.submodule_search_locations)))/'data'
    registry['independent_mtcnn_weight_sha256']={f.name:sha256(f) for f in folder.glob('*.pt')}
    write_json(paper/'model_registry.json',registry)
    if (paper/'environment.json').exists() and not (paper/'environment_setup.json').exists():
        (paper/'environment.json').rename(paper/'environment_setup.json')
    subprocess.run([sys.executable,'-m','tools.capture_research_environment','--output',str(paper/'environment.json')],check=True,cwd=ROOT)
    for name in ['environment_setup.json','environment.json']:
        path=paper/name
        if path.exists():write_json(path,redact(json.loads(path.read_text())))
    phases={name:{k:v for k,v in stage.items() if k in ['status','started_utc','finished_utc','elapsed_sec','returncode']} for name,stage in status['stages'].items()}
    write_json(paper/'provenance.json',{'run_id':run.name,'baseline_commit':'e14d89f82fe7610965513996e686739046fd90e1',
        'initial_working_tree':'clean main; no pre-existing user changes','origin':'https://github.com/jiminbae/veil',
        'processing_code_commit':processing['processing_code_commit'],'evaluation_code_commit':evaluation['evaluation_code_commit'],
        'results_commit':'resolve: git log -1 --format=%H -- paper_results/'+run.name,
        'processing_started_utc':processing['started_utc'],'processing_finished_utc':datetime.datetime.fromtimestamp((run/'primary/benchmark_summary.json').stat().st_mtime,datetime.timezone.utc).isoformat(),
        'processing_workers':1,'postprocessing_phases':phases,
        'bootstrap_seed':20261104,'inference_determinism':'no stochastic augmentation; deterministic CUDA algorithms not forced; numerical drift across environments possible',
        'historical_artifacts_sha256':{str(f.relative_to(ROOT)):sha256(f) for f in sorted((ROOT/'paper_results/panoptic20_ablation').rglob('*')) if f.is_file()},
        'analysis_source_sha256':{str(f.relative_to(ROOT)):sha256(f) for f in sorted((ROOT/'tools').glob('*.py'))},
        'raw_records_local_only':'inputs, source/output video, crops, labels, score ledgers and embeddings under ignored local_research',
        'human_verification_status':'pending; no human reviewer participated in this run',
        'transport':'original official Panoptic HTTP endpoints; HTTPS weak-certificate failure recorded',
        'independent_original_pts_spotcheck':{'clip_id':'170221_haggling_b2_hd_00_27_0001350000','first_source_frame':4046,'first_source_pts_sec':135.002,'scope':'one clip, not exhaustive timing validation'}})
    for src,name in [('tests_evaluation.log','test_results.log'),('pip_check.log','dependency_metadata_check.log')]:
        if (run/src).exists():
            (paper/name).write_text(redact((run/src).read_text()))
    figures=json.loads((paper/'figures/figure_status.json').read_text())
    completion={'overall_status':'executed_and_numerically_validated_with_pending_human_annotation_review' if not issues else 'needs_revision',
        'implementation':{'status':'tested','processing_commit':processing['processing_code_commit'],'evaluation_commit':evaluation['evaluation_code_commit']},
        'primary_processing':{'expected_runs':126,'completed_runs':summary['completed_runs'],'clips':21,'conditions':6,'status':'completed' if summary['completed_runs']==126 else 'incomplete'},
        'independent_verification':{'status':'actual_adaface_and_facenet_completed','calibration':'separate LFW empirical 1% FMR; uncertainty/dependence disclosed','stride':5},
        'numerical_result_validation':{'status':'passed' if not issues else 'failed','issues':issues},
        'human_gt_validation':{'status':'pending','reason':'No human-reviewed correction file was supplied; all projections and geometry tracks remain unverified','action':'Use annotation_review.html and export corrections with actual reviewer identity'},
        'figures':figures,'secondary_generalization':{'recordings':2,'runs':12,'annotation_level':'MTCNN geometry-track pseudo-label'},
        'historical_stress_processing':{'status':'pending','reason':'Nine rejected intervals recoverable, but original protected identities/rejected references absent; local review inputs preserved'},
        'not_generated':{'human_MOS':'No human raters','human_interrater_agreement':'No human raters',
            'pose_expression_accuracy':'No validated independent pose/expression reference',
            'flow_compensated_temporal_quality':'Not implemented; action switching is reported instead',
            'external_DeepPrivacy2_results':'Official SciPy 1.7.1 binary dependency unavailable for Python 3.12; legacy environment/port validation needed',
            'validated_low_FMR_claim':'Only empirical 1% FMR calibration; dependent pairs and domain shift',
            'public_face_images_or_videos':'Excluded by data redistribution terms and requested Git policy'},
        'git_delivery':'local branch; final commit and push outcome reported by Git after packaging',
        'scope':'No complete-anonymity, real-time, human-GT or human-quality claim'}
    write_json(paper/'completion_status.json',completion)
    if issues:raise RuntimeError('Independent raw checks found issues: '+str(issues[:5]))
    print('Packaged actual evidence; raw recomputation passed. Human annotation validation remains pending.')


if __name__=='__main__':main()
