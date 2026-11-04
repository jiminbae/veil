#!/usr/bin/env python3
"""Rebuild inspectable micro/macro tables, paired deltas and clustered intervals."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from tools.evaluate_research_run import csv_write,ratio
from tools.prepare_research_inputs import write_json

CONDITIONS=['full','no_identity_lock','no_blur_fallback','selective_blur_only','no_mouth_paste','no_swap_cache']
METRICS={
    'system_target_frame_coverage':('system_target_frame_count','expected_frames'),
    'system_nontarget_swap_fraction':('system_swap_count','system_nontarget_observations'),
    'system_nontarget_blur_fraction':('system_blur_count','system_nontarget_observations'),
    'system_nontarget_unprocessed_fraction':('system_unprocessed_count','system_nontarget_observations'),
    'derived_target_preserve_rate':('target_preserve_count','derived_target_observations'),
    'derived_target_altered_rate':(('target_swap_count','target_blur_count'),'derived_target_observations'),
    'derived_target_missed_rate':('target_missed_count','derived_target_observations'),
    'derived_nontarget_exposure_rate':(('nontarget_preserve_count','nontarget_unprocessed_count','nontarget_missed_count','nontarget_unknown_count'),'derived_nontarget_observations'),
    'derived_nontarget_false_preserve_rate':('nontarget_preserve_count','derived_nontarget_observations'),
    'derived_nontarget_missed_rate':('nontarget_missed_count','derived_nontarget_observations'),
    'system_swap_fraction':('system_swap_count','system_observations'),
    'system_blur_fraction':('system_blur_count','system_observations'),
    'derived_nontarget_swap_fraction':('nontarget_swap_count','derived_nontarget_observations'),
    'derived_nontarget_blur_fraction':('nontarget_blur_count','derived_nontarget_observations'),
    'swap_blur_switch_rate':('swap_blur_switches','swap_blur_transitions'),
}


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def number(v):return None if v in ('',None,'null','None') else float(v)


def count(row,key):
    if isinstance(key,tuple):return sum(count(row,k) for k in key)
    return number(row.get(key)) or 0.


def cluster_interval(rows,value,cluster='sequence',seed=20261104,resamples=2000):
    groups=defaultdict(list)
    for r in rows:
        v=value(r)
        if v is not None:groups[r[cluster]].append(v)
    names=sorted(groups)
    if len(names)<2:return None,None,len(names)
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(resamples):
        values=[v for g in rng.choice(names,size=len(names),replace=True) for v in groups[g]]
        samples.append(float(np.mean(values)))
    low,high=np.quantile(samples,[.025,.975])
    return float(low),float(high),len(names)


def aggregate(output,resamples=2000):
    output=Path(output);clips=read_csv(output/'per_clip_metrics.csv');runtime=read_csv(output/'runtime_metrics.csv')
    runtime_map={(r['condition'],r['clip_id']):r for r in runtime}
    rows=[r for r in clips if r['status']=='completed']
    required=set()
    for numerator,denominator in METRICS.values():
        required.update(numerator if isinstance(numerator,tuple) else [numerator])
        required.add(denominator)
    for row in rows:
        missing=[key for key in required if key not in row or number(row[key]) is None]
        if missing:
            raise ValueError(f'Cannot substitute zero for unavailable count fields: {row["clip_id"]} {missing}')
    for r in rows:
        r['recording_day']=r['sequence'].split('_')[0]
        r['end_to_end_fps']=runtime_map[(r['condition'],r['clip_id'])]['end_to_end_fps']
        for name,(num,den) in METRICS.items():
            r[name]=ratio(count(r,num),count(r,den))
    aggregates=[];intervals=[]
    for condition in CONDITIONS:
        selected=[r for r in rows if r['condition']==condition]
        clusters=len({r['sequence'] for r in selected})
        for metric,(num,den) in METRICS.items():
            numerator=sum(count(r,num) for r in selected);denominator=sum(count(r,den) for r in selected)
            values=[r[metric] for r in selected if r[metric] is not None]
            base={'condition':condition,'metric':metric,'clips':len(selected),'sequence_clusters':clusters,
                  'annotation_level':'unverified_official_3d_projection' if metric.startswith('derived') else 'system_log'}
            aggregates.append({**base,'aggregation':'micro','value':ratio(numerator,denominator),'numerator':numerator,'denominator':denominator})
            aggregates.append({**base,'aggregation':'clip_macro','value':float(np.mean(values)) if values else None,'numerator':None,'denominator':len(values)})
            for cluster in ['sequence','recording_day']:
                lo,hi,n=cluster_interval(selected,lambda r:r[metric],cluster,resamples=resamples)
                intervals.append({**base,'aggregation':'clip_macro','cluster_unit':cluster,'clusters':n,
                    'lower_95':lo,'upper_95':hi,'method':'percentile cluster bootstrap','resamples':resamples,
                    'seed':20261104,'limitation':'unknown cross-sequence person overlap; day sensitivity has only three primary clusters'})
        rt=[runtime_map[(r['condition'],r['clip_id'])] for r in selected]
        elapsed=sum(float(r['end_to_end_sec']) for r in rt)
        aggregates.append({'condition':condition,'metric':'end_to_end_fps','aggregation':'pooled_frames_over_wall_time',
            'value':ratio(sum(float(r['frames']) for r in rt),elapsed),'clips':len(rt),'numerator':sum(float(r['frames']) for r in rt),'denominator':elapsed})
    paired=[];sensitivity=[]
    lookup={(r['condition'],r['clip_id']):r for r in rows}
    for comparison in CONDITIONS[1:]:
        for metric in ['derived_target_preserve_rate','derived_nontarget_exposure_rate','end_to_end_fps','swap_blur_switch_rate']:
            pairs=[]
            for full in [r for r in rows if r['condition']=='full']:
                other=lookup.get((comparison,full['clip_id']))
                if other is None:continue
                av=number(full.get(metric));bv=number(other.get(metric))
                if av is None or bv is None:continue
                pair={'reference':'full','comparison':comparison,'clip_id':full['clip_id'],'sequence':full['sequence'],
                    'recording_day':full['recording_day'],'metric':metric,'reference_value':av,'comparison_value':bv,'delta_full_minus_comparison':av-bv}
                pairs.append(pair);paired.append(pair)
            for cluster in ['sequence','recording_day']:
                lo,hi,n=cluster_interval(pairs,lambda r:r['delta_full_minus_comparison'],cluster,resamples=resamples)
                intervals.append({'condition':'full - '+comparison,'metric':metric,'aggregation':'paired_clip_macro',
                    'cluster_unit':cluster,'clusters':n,'clips':len(pairs),'lower_95':lo,'upper_95':hi,
                    'estimate':float(np.mean([r['delta_full_minus_comparison'] for r in pairs])) if pairs else None,
                    'method':'paired percentile cluster bootstrap','resamples':resamples,'seed':20261104})
            for omitted in sorted({r['sequence'] for r in pairs}):
                values=[r['delta_full_minus_comparison'] for r in pairs if r['sequence']!=omitted]
                sensitivity.append({'comparison':comparison,'metric':metric,'omitted_sequence':omitted,
                    'remaining_clips':len(values),'mean_delta':float(np.mean(values)) if values else None})
    verifier_path=output/'independent_identity_verifier/per_clip_verifier_metrics.csv'
    if verifier_path.exists():
        verifier=read_csv(verifier_path)
        for condition in CONDITIONS:
            for backend in ['adaface','facenet']:
                for group in ['target','non_target','non_target_swap','non_target_blur']:
                    selected=[r for r in verifier if r['condition']==condition and r['backend']==backend and r['group']==group]
                    for metric,num,den in [
                        ('accept_rate_successful_pairs','accepted','score_successes'),
                        ('accept_rate_lower_bound_all_attempts','accepted','attempts'),
                        ('original_baseline_accept_rate','original_baseline_accepted','original_baseline_scored'),
                        ('source_extraction_success_rate','source_embedding_successes','attempts'),
                        ('output_extraction_success_rate','output_embedding_successes','attempts'),
                        ('original_verified_subset_accept_rate','original_verified_subset_accepted','original_verified_subset_scored')]:
                        numerator=sum(count(r,num) for r in selected);denominator=sum(count(r,den) for r in selected)
                        aggregates.append({'condition':condition,'metric':metric,'backend':backend,'group':group,
                            'aggregation':'micro','value':ratio(numerator,denominator),'numerator':numerator,'denominator':denominator,
                            'clips':len(selected),'annotation_level':'unverified_projection; calibrated real verifier'})
                        if metric=='accept_rate_successful_pairs':
                            values=[number(r.get(metric)) for r in selected];values=[v for v in values if v is not None]
                            aggregates.append({'condition':condition,'metric':metric,'backend':backend,'group':group,
                                'aggregation':'clip_macro','value':float(np.mean(values)) if values else None,
                                'denominator':len(values),'numerator':None,'clips':len(selected)})
                            lo,hi,n=cluster_interval(selected,lambda r:number(r.get(metric)),resamples=resamples)
                            intervals.append({'condition':condition,'metric':metric,'backend':backend,'group':group,
                                'aggregation':'clip_macro','cluster_unit':'sequence','clusters':n,'lower_95':lo,'upper_95':hi,
                                'method':'percentile cluster bootstrap','resamples':resamples,'seed':20261104})
                    accepted=sum(count(r,'accepted') for r in selected);attempts=sum(count(r,'attempts') for r in selected)
                    failures=sum(count(r,'score_failures') for r in selected)
                    aggregates.append({'condition':condition,'metric':'accept_rate_upper_bound_all_attempts',
                        'backend':backend,'group':group,'aggregation':'micro','value':ratio(accepted+failures,attempts),
                        'numerator':accepted+failures,'denominator':attempts,'clips':len(selected)})
        verifier_lookup={(r['condition'],r['clip_id'],r['backend'],r['group']):r for r in verifier}
        for comparison in CONDITIONS[1:]:
            for backend in ['adaface','facenet']:
                for group in ['target','non_target']:
                    for metric in ['accept_rate_successful_pairs','accept_rate_lower_bound_all_attempts','accept_rate_upper_bound_all_attempts']:
                        pairs=[]
                        for full in [r for r in verifier if r['condition']=='full' and r['backend']==backend and r['group']==group]:
                            other=verifier_lookup.get((comparison,full['clip_id'],backend,group))
                            if other is None:continue
                            av=number(full.get(metric));bv=number(other.get(metric))
                            if av is None or bv is None:continue
                            pair={'reference':'full','comparison':comparison,'clip_id':full['clip_id'],'sequence':full['sequence'],
                                'backend':backend,'group':group,'metric':metric,'reference_value':av,'comparison_value':bv,
                                'delta_full_minus_comparison':av-bv,'reference_attempts':full['attempts'],
                                'comparison_attempts':other['attempts'],'reference_scored':full['score_successes'],
                                'comparison_scored':other['score_successes'],
                                'denominator_note':'same clips; conditional score subsets can differ; all-attempt bounds retain extraction failures'}
                            pairs.append(pair);paired.append(pair)
                        lo,hi,n=cluster_interval(pairs,lambda r:r['delta_full_minus_comparison'],resamples=resamples)
                        intervals.append({'condition':'full - '+comparison,'metric':metric,'backend':backend,'group':group,
                            'aggregation':'paired_clip_macro','cluster_unit':'sequence','clusters':n,'clips':len(pairs),
                            'lower_95':lo,'upper_95':hi,'estimate':float(np.mean([r['delta_full_minus_comparison'] for r in pairs])) if pairs else None,
                            'method':'paired percentile cluster bootstrap','resamples':resamples,'seed':20261104})
                        for omitted in sorted({r['sequence'] for r in pairs}):
                            values=[r['delta_full_minus_comparison'] for r in pairs if r['sequence']!=omitted]
                            sensitivity.append({'comparison':comparison,'backend':backend,'group':group,'metric':metric,
                                'omitted_sequence':omitted,'remaining_clips':len(values),'mean_delta':float(np.mean(values)) if values else None})
        people=read_csv(output/'independent_identity_verifier/per_identity_verifier_metrics.csv')
        for condition in CONDITIONS:
            for backend in ['adaface','facenet']:
                for target in [True,False]:
                    selected=[r for r in people if r['condition']==condition and r['backend']==backend and (r['is_target']=='True')==target]
                    for key in ['any_frame_accepted','any_frame_risk_upper_bound','mean_embedding_attack_accepted']:
                        valid=[r for r in selected if r[key]!='']
                        numerator=sum(r[key]=='True' for r in valid)
                        aggregates.append({'condition':condition,'backend':backend,'group':'target' if target else 'non_target',
                            'metric':key,'aggregation':'person_clip','value':ratio(numerator,len(valid)),
                            'numerator':numerator,'denominator':len(valid),'clips':len({r['clip_id'] for r in valid})})
    csv_write(output/'aggregate_metrics.csv',aggregates);write_json(output/'aggregate_metrics.json',aggregates)
    csv_write(output/'paired_comparisons.csv',paired);csv_write(output/'confidence_intervals.csv',intervals)
    csv_write(output/'leave_one_sequence_out.csv',sensitivity)
    write_json(output/'aggregation_audit.json',{'unique_per_clip_keys':len({(r['condition'],r['clip_id']) for r in clips})==len(clips),
        'failed_rows_retained':sum(r['status']!='completed' for r in clips),'completed_rows':len(rows),
        'primary_cluster_unit':'sequence','bootstrap_seed':20261104,'resamples':resamples,
        'recording_day_sensitivity':True,'source_files':['per_clip_metrics.csv','runtime_metrics.csv','independent_identity_verifier/per_clip_verifier_metrics.csv']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--resamples',type=int,default=2000)
    a=p.parse_args();aggregate(a.output_dir,a.resamples)
