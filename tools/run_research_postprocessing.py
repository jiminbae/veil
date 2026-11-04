#!/usr/bin/env python3
"""Durable, restartable postprocessing after the frozen single-worker matrix.

The controller uses only the current run's local files. It does not publish,
commit, install packages or alter other processes. Completed stages require the
same command/code/input signature before reuse.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tools.prepare_research_inputs import ROOT,write_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--paper-dir',type=Path,required=True);p.add_argument('--wait-for-primary',action='store_true')
    a=p.parse_args();run=a.run_root.resolve();paper=a.paper_dir.resolve();paper.mkdir(parents=True,exist_ok=True)
    status_path=run/'postprocessing_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {'stages':{},'status':'running'}
    code={str(path.relative_to(ROOT)):sha256(path) for path in sorted((ROOT/'tools').glob('*.py'))}
    job=json.loads((run/'primary_job.json').read_text())
    summary_path=run/'primary/benchmark_summary.json'
    while not summary_path.exists():
        if not a.wait_for_primary:raise RuntimeError('Primary matrix is not finished')
        try:os.kill(job['pid'],0)
        except ProcessLookupError:raise RuntimeError('Primary exited without a final summary; inspect primary.log')
        time.sleep(3)
    primary=json.loads(summary_path.read_text())
    if primary['total_runs']!=126 or primary['completed_runs']!=126 or primary['failed_runs']:
        raise RuntimeError('Primary has incomplete/failed runs; preserve ledger and inspect before continuing')
    signature={'tools':code,'input_manifest':sha256(run/'inputs/dataset_manifest.jsonl'),
        'primary_summary':sha256(summary_path),'calibration':{b:sha256(run/'calibration/results'/(b+'_calibration.json')) for b in ['adaface','facenet']}}
    def execute(name,module,args,required):
        if any(sha256(ROOT/path)!=digest for path,digest in code.items()):
            raise RuntimeError('Evaluation source changed after controller startup; preserve stages and start a new evaluation attempt')
        command=[sys.executable,'-u','-m',module,*map(str,args)]
        fingerprint=hashlib.sha256(json.dumps({'command':command,'signature':signature},sort_keys=True).encode()).hexdigest()
        previous=status['stages'].get(name)
        if previous and previous['status']=='completed':
            if previous['fingerprint']!=fingerprint:raise RuntimeError(f'Stage {name} code/input mismatch; use a new output directory')
            if not all(Path(v).exists() and sha256(v)==h for v,h in previous['outputs'].items()):raise RuntimeError(f'Stage {name} output integrity mismatch')
            print(name,'reused verified stage',flush=True);return
        started=time.monotonic();stage={'status':'running','fingerprint':fingerprint,'command':command,
            'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        status['stages'][name]=stage;write_json(status_path,status)
        print(name,'started',flush=True)
        with (run/(name+'.log')).open('a') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),check=False)
        stage.update(elapsed_sec=time.monotonic()-started,returncode=result.returncode,
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if result.returncode or not all(Path(path).exists() for path in required):
            stage['status']='failed';write_json(status_path,status)
            raise RuntimeError(f'{name} failed; inspect its log. No automatic endless retry.')
        stage.update(status='completed',outputs={str(path):sha256(path) for path in required})
        write_json(status_path,status);print(name,'completed',round(stage['elapsed_sec'],1),'sec',flush=True)
    supplement=run/'supplement'
    if (supplement/'inputs/dataset_manifest.jsonl').exists():
        saved=supplement/'benchmark/benchmark_summary.json'
        if saved.exists():
            check=json.loads(saved.read_text())
            attempts=json.loads((supplement/'benchmark/run_summary.json').read_text())
            if check['total_runs']!=12 or len(attempts)!=12:
                raise RuntimeError('Secondary attempt ledger is incomplete; preserve and inspect it')
            secondary_rows={r['clip_id']:r for r in (json.loads(s) for s in (supplement/'inputs/dataset_manifest.jsonl').read_text().splitlines())}
            model_candidates=list((ROOT/'models/veil/weights').glob('*'))+list((Path.home()/'.insightface/models/buffalo_l').glob('*.onnx'))
            model_hashes={p.name:sha256(p) for p in model_candidates if p.is_file()}
            for attempt in attempts:
                if attempt['status'] not in {'completed','skipped_existing'}:continue
                directory=Path(attempt['face_metadata']).parent
                receipt=json.loads((directory/'run_receipt.json').read_text())
                fp=receipt['fingerprint'];row=secondary_rows[attempt['clip_id']]
                if fp['clip']!=sha256(ROOT/row['clip_path']) or fp['protected_reference']!=sha256(ROOT/row['target_image_path']) or fp['replacement']!=sha256(ROOT/'media/fake_face.jpg'):
                    raise RuntimeError('Secondary input fingerprint mismatch')
                if fp['models']!=model_hashes or any(sha256(ROOT/path)!=digest for path,digest in fp['code'].items()):
                    raise RuntimeError('Secondary model/processing-code fingerprint mismatch')
                if not all((directory/name).exists() and sha256(directory/name)==digest for name,digest in receipt['outputs'].items()):
                    raise RuntimeError('Secondary completed output hash mismatch')
            previous=status['stages'].get('supplement_processing',{})
            status['stages']['supplement_processing']={**previous,
                'status':'completed_with_failed_inputs' if check['failed_runs'] else 'completed',
                'completed_runs':check['completed_runs'],'failed_runs':check['failed_runs'],
                'adoption_note':'Preserved original attempts. Failed target registration is a system/input failure, not a reason to discard the clip or block primary analysis.',
                'outputs':{str(saved):sha256(saved)}}
            write_json(status_path,status)
        else:
            execute('supplement_processing','tools.run_veil_benchmark',[
                '--manifest',supplement/'inputs/dataset_manifest.jsonl','--output-dir',supplement/'benchmark',
                '--replacement-image',ROOT/'media/fake_face.jpg','--conditions','full','no_identity_lock',
                'no_blur_fallback','selective_blur_only','no_mouth_paste','no_swap_cache',
                '--python',sys.executable,'--timeout-sec','900','--skip-existing'],[saved])
        check=json.loads((supplement/'benchmark/benchmark_summary.json').read_text())
        print('Secondary processing:',check['completed_runs'],'completed,',check['failed_runs'],'failed; all attempts retained',flush=True)
    for label,base,benchmark,dest in [('primary',run,run/'primary',paper),
        ('supplement',supplement,supplement/'benchmark',paper/'chokepoint_supplement')]:
        if not (base/'inputs/dataset_manifest.jsonl').exists():continue
        local=base/'evaluation'
        execute(label+'_actions','tools.evaluate_research_run',[
            '--manifest',base/'inputs/dataset_manifest.jsonl','--annotations',base/'annotations',
            '--benchmark',benchmark,'--output-dir',dest,'--local-output',local],
            [dest/'per_clip_metrics.csv',dest/'per_identity_metrics.csv',dest/'runtime_metrics.csv',dest/'integrity_checks.json'])
        execute(label+'_verification','tools.score_research_identity',[
            '--manifest',base/'inputs/dataset_manifest.jsonl','--annotations',base/'annotations',
            '--benchmark',benchmark,'--calibration',run/'calibration/results',
            '--output-dir',dest/'independent_identity_verifier','--local-output',local,
            '--adaface-model',ROOT/'models/veil/weights/adaface_ir50_ms1mv2.ckpt'],
            [dest/'independent_identity_verifier/per_clip_verifier_metrics.csv',dest/'independent_identity_verifier/per_identity_verifier_metrics.csv',dest/'independent_identity_verifier/evaluation_status.json'])
        execute(label+'_utility','tools.evaluate_research_utility',[
            '--manifest',base/'inputs/dataset_manifest.jsonl','--annotations',base/'annotations',
            '--benchmark',benchmark,'--controls',local/'controls','--output',dest/'quality_metrics.csv'],[dest/'quality_metrics.csv'])
        if label=='primary':
            execute('primary_aggregate','tools.aggregate_research_results',['--output-dir',dest],
                [dest/'aggregate_metrics.csv',dest/'aggregate_metrics.json',dest/'paired_comparisons.csv',dest/'confidence_intervals.csv'])
            execute('primary_figures','tools.plot_research_results',['--output-dir',dest],
                [dest/'figures/figure_status.json'])
    status.update(status='completed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(status_path,status)


if __name__=='__main__':main()
