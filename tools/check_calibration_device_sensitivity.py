#!/usr/bin/env python3
"""Repeat calibration without replacing the pre-frozen evaluation thresholds."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tools.prepare_research_inputs import ROOT,write_json,sha256
from tools.aggregate_research_results import read_csv


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--paper-dir',type=Path,required=True)
    p.add_argument('--wait-for-postprocessing',action='store_true');a=p.parse_args();run=a.run_root.resolve();paper=a.paper_dir.resolve()
    if a.wait_for_postprocessing:
        while True:
            path=run/'postprocessing_status.json'
            if path.exists():
                status=json.loads(path.read_text())
                if status['status']=='completed':break
                if any(v['status']=='failed' for v in status['stages'].values()):raise RuntimeError('Postprocessing failed; do not add GPU workloads')
            time.sleep(3)
    output=run/'calibration/revalidation_gpu'
    code_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'tools/calibrate_independent_verifiers.py',ROOT/'tools/independent_verifiers.py',ROOT/'local_research/vendor/adaface_net.py']}
    if output.exists():raise RuntimeError('Use a fresh validation attempt; do not overwrite calibration evidence')
    started=datetime.datetime.now(datetime.timezone.utc)
    with (run/'calibration_revalidation_gpu.log').open('x') as log:
        subprocess.run([sys.executable,'-u','-m','tools.calibrate_independent_verifiers',
            '--lfw-archive',str(run/'calibration/lfw.tgz'),'--output-dir',str(output),
            '--adaface-model',str(ROOT/'models/veil/weights/adaface_ir50_ms1mv2.ckpt')],
            check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    rows=[]
    for backend in ['adaface','facenet']:
        original=json.loads((run/'calibration/results'/(backend+'_calibration.json')).read_text())
        for label,folder in [('repeated_cuda',output),('cpu',run/'calibration/revalidation_cpu')]:
            result=json.loads((folder/(backend+'_calibration.json')).read_text())
            scores=read_csv(folder/(backend+'_pair_scores.csv'))
            negatives=[float(r['score']) for r in scores if r['same_identity']=='False']
            threshold=original['threshold'];fp=sum(v>=threshold for v in negatives)
            rows.append({'backend':backend,'comparison':label,'frozen_cuda_threshold':threshold,
                'repeat_device':result['model']['device'],'repeat_calibration_threshold':result['threshold'],
                'threshold_delta':result['threshold']-threshold,'same_checkpoint':original['model']['checkpoint_sha256']==result['model']['checkpoint_sha256'],
                'same_selection':original['selection_sha256']==result['selection_sha256'],
                'frozen_threshold_false_accepts_on_repeat':fp,'impostor_pairs':len(negatives),
                'frozen_threshold_empirical_fmr_on_repeat':fp/len(negatives),
                'repeat_genuine_acceptance_at_its_calibration_threshold':result['genuine_acceptance']})
    write_json(paper/'verifier_calibration/device_sensitivity.json',{
        'started_utc':started.isoformat(),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'code_sha256':code_hashes,'comparisons':rows,'original_thresholds_replaced':False,
        'same_device_threshold_reproduction_within_1e_6':all(abs(r['threshold_delta'])<1e-6 for r in rows if r['comparison']=='repeated_cuda'),
        'interpretation':'Device sensitivity audit only. Keep the original pre-test CUDA thresholds; CPU and CUDA operating points are not interchangeable. This is not test-video threshold tuning.'})
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
