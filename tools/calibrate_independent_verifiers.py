#!/usr/bin/env python3
"""Freeze verifier thresholds on a separate, deterministically selected corpus."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import tarfile
import numpy as np
from PIL import Image
from scipy.stats import beta
from tools.independent_verifiers import FaceRecognitionVerifier,file_hash
from tools.prepare_research_inputs import write_json


def calibrate_scores(genuine,impostor,fmr=.01):
    impostor=np.asarray(impostor,float);genuine=np.asarray(genuine,float)
    if len(impostor)==0 or len(genuine)==0:raise ValueError('Both pair classes are required')
    # Strictly above an observed impostor quantile; ties cannot violate the bound.
    ordered=np.sort(impostor)
    idx=max(0,int(np.ceil((1-fmr)*len(ordered)))-1)
    threshold=float(np.nextafter(ordered[idx],float('inf')))
    fp=int((impostor>=threshold).sum());n=len(impostor)
    interval=[float(beta.ppf(.025,fp,n-fp+1)) if fp else 0.,
              float(beta.ppf(.975,fp+1,n-fp)) if fp<n else 1.]
    return {'threshold':threshold,'requested_empirical_fmr':fmr,'impostor_pairs':n,
        'false_accepts':fp,'empirical_fmr':fp/n,'genuine_pairs':len(genuine),
        'genuine_acceptance':float((genuine>=threshold).mean()),
        'descriptive_pair_binomial_interval_95':interval,
        'interval_limitation':'Pairs share identities; binomial interval is descriptive, not an independent-pair population guarantee. No low-FMR generalization claim.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--lfw-archive',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--identities',type=int,default=100)
    p.add_argument('--adaface-model',type=Path,required=True);args=p.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    # Read JPEGs directly; never extract arbitrary archive paths.
    with tarfile.open(args.lfw_archive,'r:gz') as archive:
        names={}
        for member in archive.getmembers():
            if member.isfile() and member.name.lower().endswith('.jpg'):
                names.setdefault(Path(member.name).parent.name,[]).append(member.name)
        subjects=[(name,sorted(files)[:2]) for name,files in sorted(names.items()) if len(files)>=2][:args.identities]
        selection=[{'identity':name,'files':files} for name,files in subjects]
        write_json(out/'selection.json',{'source':'https://ndownloader.figshare.com/files/5976018',
            'archive_sha256':file_hash(args.lfw_archive),'selection_rule':'first 100 alphabetical identities with >=2 images; first two filenames',
            'subjects':selection,'test_corpus':'Panoptic; no shared sequences; no global person registry to prove identity disjointness',
            'license_note':'LFW research corpus; source images excluded from redistribution'})
        for backend in ['adaface','facenet']:
            path=out/(backend+'_calibration.json')
            if path.exists():
                print(backend,'already frozen',flush=True);continue
            model=FaceRecognitionVerifier(backend,args.adaface_model if backend=='adaface' else None)
            embeddings=[];attempts=[]
            for name,files in subjects:
                pair=[]
                for filename in files:
                    img=Image.open(archive.extractfile(filename)).convert('RGB')
                    emb=model.extract_embedding(img)
                    pair.append(emb)
                    attempts.append({'identity':name,'file':filename,'success':emb is not None,'failure':model.last_failure})
                embeddings.append(pair)
            genuine=[];impostor=[];scores=[]
            for i,(name,_) in enumerate(subjects):
                for j,(other,_) in enumerate(subjects):
                    a,b=embeddings[i][0],embeddings[j][1]
                    if a is None or b is None:continue
                    score=model.similarity(a,b)
                    (genuine if i==j else impostor).append(score)
                    scores.append({'enrollment_id':i,'probe_id':j,'same_identity':i==j,'score':score})
            result={**calibrate_scores(genuine,impostor),'backend':backend,
                'attempted_images':2*len(subjects),'embedding_successes':sum(a['success'] for a in attempts),
                'subjects':len(subjects),'model':model.metadata,
                'selection_sha256':file_hash(out/'selection.json'),'frozen_before_test_scoring':True,
                'status':'calibrated_on_separate_corpus_empirical_operating_point'}
            write_json(path,result);write_json(out/(backend+'_attempts.json'),attempts)
            with (out/(backend+'_pair_scores.csv')).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(scores[0]));w.writeheader();w.writerows(scores)
            print(backend,'threshold',result['threshold'],'fmr',result['empirical_fmr'],
                  'genuine',result['genuine_acceptance'],'successes',result['embedding_successes'],flush=True)
            del model


if __name__=='__main__':main()
