#!/usr/bin/env python3
"""Capture shareable environment details; avoid credentials and personal paths."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(args):
    r=subprocess.run(args,text=True,capture_output=True,check=False)
    return {'returncode':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    usage=shutil.disk_usage(root)
    env={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'os':platform.platform(),'python':sys.version,'cpu_logical_count':os.cpu_count(),
         'cpu':command(['lscpu']), 'ram':command(['free','-b']),
         'disk_bytes':{'total':usage.total,'used':usage.used,'free':usage.free},
         'gpu':command(['nvidia-smi','--query-gpu=name,memory.total,driver_version,memory.used,utilization.gpu','--format=csv,noheader']),
         'gpu_processes':command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']),
         'git_head':command(['git','rev-parse','HEAD']),
         'git_worktree':command(['git','status','--porcelain']),
         'ffmpeg':command(['ffmpeg','-version'])}
    env['packages']={}
    for name in ['torch','torchvision','numpy','scipy','pillow','opencv-python','onnxruntime-gpu','insightface','ultralytics','boxmot','facenet-pytorch','matplotlib','pytest']:
        try: env['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: env['packages'][name]=None
    try:
        import torch
        env['torch_runtime']={'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
            'device_count':torch.cuda.device_count(),'device_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    except Exception as e: env['torch_runtime']={'error':str(e)}
    try:
        import onnxruntime as ort
        env['onnxruntime']={'version':ort.__version__,'available_providers':ort.get_available_providers(),
            'actual_providers':'recorded per-session in each completed runtime.json; availability alone is not actual usage'}
    except Exception as e: env['onnxruntime']={'error':str(e)}
    # CPU list/ffmpeg build are public environment specifications. Process arguments
    # and environment variables are intentionally not captured.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(env,indent=2)+'\n')


if __name__=='__main__':main()
