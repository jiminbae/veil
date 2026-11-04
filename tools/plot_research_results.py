#!/usr/bin/env python3
"""Export source-backed standalone PNG/SVG research figures."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from tools.aggregate_research_results import CONDITIONS,read_csv,number
from tools.prepare_research_inputs import write_json
from tools.evaluate_research_run import csv_write

BLUE='#295D9C';ORANGE='#D88B22';GRAY='#8F9DAA';DARK='#23303D'
SHORT={'full':'full','no_identity_lock':'no identity\nlock','no_blur_fallback':'no blur\nfallback',
       'selective_blur_only':'selective\nblur only','no_mouth_paste':'no mouth\npaste','no_swap_cache':'no swap\ncache'}


def plot(output):
    output=Path(output);figdir=output/'figures';figdir.mkdir(exist_ok=True)
    rows=read_csv(output/'aggregate_metrics.csv');ci=read_csv(output/'confidence_intervals.csv')
    clips=read_csv(output/'per_clip_metrics.csv');runtime=read_csv(output/'runtime_metrics.csv')
    x=np.arange(len(CONDITIONS));labels=[SHORT[c] for c in CONDITIONS];generated=[];omitted={}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelcolor':DARK,'text.color':DARK,'axes.titleweight':'bold',
        'svg.fonttype':'none','savefig.facecolor':'white'})
    nclips=len({r['clip_id'] for r in clips});nseq=len({r['sequence'] for r in clips})
    scope=f'Panoptic reconstructed {nclips} clips; {nseq} sequences. Projected labels are unverified; visibility eligibility applies.'
    def value(condition,metric,aggregation='micro',backend='',group=''):
        candidates=[r for r in rows if r['condition']==condition and r['metric']==metric
            and r['aggregation']==aggregation and r.get('backend','')==backend and r.get('group','')==group]
        return number(candidates[0]['value']) if candidates else None
    def save(fig,name,note,top=1):
        fig.text(.02,.015,note,fontsize=9,ha='left',va='bottom',wrap=True)
        fig.tight_layout(rect=(0,.10,1,top))
        for suffix in ['png','svg']:fig.savefig(figdir/(name+'.'+suffix),dpi=180,bbox_inches='tight')
        plt.close(fig);generated.append(name)
    fig,ax=plt.subplots(figsize=(12,5.8))
    for k,(metric,label,color) in enumerate([
        ('derived_target_preserve_rate','Target preserved',BLUE),
        ('derived_nontarget_exposure_rate','Non-target potential exposure',ORANGE)]):
        values=[value(c,metric,'clip_macro') for c in CONDITIONS]
        if any(v is None for v in values):continue
        low=[];high=[]
        for c,v in zip(CONDITIONS,values):
            match=next((r for r in ci if r['condition']==c and r['metric']==metric and r.get('cluster_unit')=='sequence'),None)
            lo=number(match['lower_95']) if match else None;hi=number(match['upper_95']) if match else None
            low.append(v-lo if lo is not None else 0);high.append(hi-v if hi is not None else 0)
        ax.bar(x+(k-.5)*.34,values,width=.34,color=color,label=label,
            yerr=np.maximum(0,np.array([low,high])),capsize=3)
    ax.set(xticks=x,xticklabels=labels,ylim=(0,1.12),ylabel='Clip-macro face-observation rate',
        title='Projected-label target preservation and potential non-target exposure')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(loc='upper center',ncol=2,frameon=False)
    save(fig,'target_preservation_and_exposure',scope+'\nWhiskers: 95% sequence-cluster bootstrap. Valid clip denominators are in aggregate_metrics.csv. Exposure includes missed/unknown faces, not re-identification.')
    fig,ax=plt.subplots(figsize=(12,5.8));bottom=np.zeros(len(x))
    for metric,label,color,hatch in [('derived_nontarget_swap_fraction','Swap',BLUE,None),('derived_nontarget_blur_fraction','Blur',ORANGE,'//')]:
        values=np.array([value(c,metric) for c in CONDITIONS],dtype=float);ax.bar(x,values,bottom=bottom,color=color,label=label,hatch=hatch);bottom+=values
    ax.bar(x,1-bottom,bottom=bottom,color=GRAY,label='Preserved / missed / unprocessed / unknown')
    ax.set(xticks=x,xticklabels=labels,ylim=(0,1.15),ylabel='Micro fraction of eligible projected non-target faces',title='Swap and blur usage on independently projected non-target faces')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(loc='upper center',ncol=3,frameon=False,fontsize=10)
    save(fig,'swap_blur_usage',scope+'\nDenominator: all camera-visible non-target projected face observations; missed observations remain in the denominator.')
    if (output/'independent_identity_verifier/per_clip_verifier_metrics.csv').exists():
        fig,axes=plt.subplots(1,2,figsize=(15,6),sharey=True)
        for ax,backend,color in zip(axes,['adaface','facenet'],[BLUE,ORANGE]):
            lower=[value(c,'accept_rate_lower_bound_all_attempts',backend=backend,group='non_target') for c in CONDITIONS]
            upper=[value(c,'accept_rate_upper_bound_all_attempts',backend=backend,group='non_target') for c in CONDITIONS]
            scored=[value(c,'accept_rate_successful_pairs',backend=backend,group='non_target') for c in CONDITIONS]
            ax.vlines(x,lower,upper,color=color,linewidth=7,alpha=.45,label='All-attempt risk bounds')
            ax.scatter(x,scored,color=DARK,marker='x',s=65,label='Acceptance among scored pairs',zorder=3)
            ax.set(xticks=x,xticklabels=labels,ylim=(0,1.04),title=backend,ylabel='Original-identity acceptance')
            ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False,fontsize=9)
        fig.suptitle('Independent verifier re-identification: extraction failures remain unknown',fontsize=15,fontweight='bold')
        save(fig,'independent_verifier_reidentification',scope+'\nEvery fifth frame. Bars span accepted/all attempts to (accepted+unknown)/all attempts, not statistical CIs. LFW empirical FMR=1%.')
    else:omitted['independent_verifier_reidentification']='Real-model scoring not yet available'
    paired=read_csv(output/'paired_comparisons.csv')
    effects=[r for r in paired if r['comparison']=='no_identity_lock' and r['metric']=='derived_target_preserve_rate']
    effects.sort(key=lambda r:number(r['delta_full_minus_comparison']))
    if effects:
        fig,ax=plt.subplots(figsize=(14,max(7,.36*len(effects))))
        vals=[100*number(r['delta_full_minus_comparison']) for r in effects]
        ax.barh(np.arange(len(effects)),vals,color=BLUE)
        ax.axvline(0,color=DARK,linewidth=1)
        ax.set(yticks=np.arange(len(effects)),yticklabels=[r['clip_id'] for r in effects],
            xlabel='Target preserve rate: full minus no_identity_lock (percentage points)',
            title='Identity-lock effect by clip under unverified projected labels')
        save(fig,'identity_lock_per_clip',scope+'\nPaired clips; embedding cache and tracking remain active in no_identity_lock. Zero means no observed difference.')
    quality_path=output/'quality_metrics.csv'
    if quality_path.exists() and (output/'independent_identity_verifier/per_clip_verifier_metrics.csv').exists():
        quality=read_csv(quality_path);fig,axes=plt.subplots(1,2,figsize=(15,6),sharey=True)
        verifier=read_csv(output/'independent_identity_verifier/per_clip_verifier_metrics.csv')
        quality_map={(r['condition'],r['clip_id']):r for r in quality};source_rows=[]
        for ax,backend,color in zip(axes,['adaface','facenet'],[BLUE,ORANGE]):
            for idx,c in enumerate(CONDITIONS):
                pairs=[]
                for r in verifier:
                    if r['condition']!=c or r['backend']!=backend or r['group']!='non_target':continue
                    q=quality_map.get((c,r['clip_id']))
                    if q is None:continue
                    mae=number(q['excess_mae_0_255']);risk=number(r['accept_rate_successful_pairs'])
                    if mae is not None and risk is not None:pairs.append((r['clip_id'],mae,risk))
                if pairs:
                    xx=float(np.mean([p[1] for p in pairs]));risk=float(np.mean([p[2] for p in pairs]))
                    ax.scatter([xx],[risk],color=color,marker=['o','s','^','D','v','P'][idx],s=65,label=c.replace('_',' '))
                    source_rows.append({'backend':backend,'condition':c,'paired_clips':len(pairs),
                        'target_excess_mae':xx,'conditional_acceptance':risk,'clip_ids':json.dumps([p[0] for p in pairs])})
            ax.set(xlabel='Target ROI excess MAE over same-codec control (0–255)',ylabel='Clip-macro acceptance among scored non-target pairs',ylim=(-.03,1.12),title=backend)
            ax.yaxis.set_major_formatter(PercentFormatter(1));ax.margins(x=.25)
        handles,names=axes[0].get_legend_handles_labels()
        fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,frameon=False,fontsize=10)
        fig.suptitle('Target pixel preservation and conditional re-identification',fontsize=15,fontweight='bold')
        csv_write(figdir/'utility_vs_risk_source.csv',source_rows)
        save(fig,'utility_vs_reidentification',scope+'\nBoth axes use the same valid clips; counts in utility_vs_risk_source.csv. Coincident markers overlap. MAE is not perceptual quality; verifier excludes extraction failures.',top=.80)
    else:omitted['utility_vs_reidentification']='Quality or real-verifier measurements unavailable'
    fig,ax=plt.subplots(figsize=(12,5.8))
    data=[[float(r['end_to_end_fps']) for r in runtime if r['condition']==c] for c in CONDITIONS]
    ax.boxplot(data,positions=x,widths=.5,patch_artist=True,
        boxprops={'facecolor':'#DDE7F2','edgecolor':BLUE},medianprops={'color':DARK},showfliers=True)
    pooled=[value(c,'end_to_end_fps','pooled_frames_over_wall_time') for c in CONDITIONS]
    ax.scatter(x,pooled,color=ORANGE,marker='D',label='Pooled frames / subprocess wall time',zorder=3)
    ax.set(xticks=x,xticklabels=labels,ylim=(0,None),ylabel='Frames per second',title='End-to-end processing speed, including subprocess startup')
    ax.legend(frameon=False)
    save(fig,'condition_processing_speed',f'Panoptic {nclips} clips; 1920×1080; one worker; swap batch=8. RTX PRO 6000 Blackwell 96GB.\nBoxes show per-clip FPS distribution, diamonds show pooled FPS. Includes model imports/loading/encoding; no real-time claim.')
    failures=read_csv(output/'failure_cases.csv');categories=sorted({r['failure_type'] for r in failures if not r['failure_type'].startswith('system_reason_')})
    matrix=np.array([[sum(int(r['count']) for r in failures if r['condition']==c and r['failure_type']==cat) for cat in categories] for c in CONDITIONS])
    if categories:
        fig,ax=plt.subplots(figsize=(max(10,1.5*len(categories)),6))
        image=ax.imshow(matrix,aspect='auto',cmap='Blues',vmin=0)
        ax.set(yticks=np.arange(len(CONDITIONS)),yticklabels=CONDITIONS,xticks=np.arange(len(categories)),
            xticklabels=[c.replace('nontarget_','non-target\n').replace('target_','target\n') for c in categories],
            title='Failure-state counts under projected labels')
        for i in range(len(CONDITIONS)):
            for j in range(len(categories)):
                ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>.6*matrix.max() else DARK)
        fig.colorbar(image,ax=ax,label='Eligible face observations')
        save(fig,'failure_type_distribution',scope+'\nStates are observation counts, not independent people. Missed / unknown non-target faces indicate potential exposure.')
    else:omitted['failure_type_distribution']='No observed failure-state rows'
    write_json(figdir/'figure_status.json',{'generated':generated,'omitted':omitted,
        'formats':['PNG','SVG'],'source_files':['aggregate_metrics.csv','confidence_intervals.csv','paired_comparisons.csv','runtime_metrics.csv','quality_metrics.csv','failure_cases.csv'],
        'validation':'numerical source checks and rendered visual inspection are recorded separately'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();plot(a.output_dir)
