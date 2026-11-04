#!/usr/bin/env python3
"""Write the evidence guide and final lightweight delivery receipts."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import re
import subprocess
from tools.prepare_research_inputs import ROOT,write_json
from tools.evaluate_research_run import csv_write
from tools.aggregate_research_results import read_csv,number


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--paper-dir',type=Path,required=True);a=p.parse_args()
    run=a.run_root.resolve();paper=a.paper_dir.resolve()
    completion=json.loads((paper/'completion_status.json').read_text())
    if completion['numerical_result_validation']['status']!='passed':raise RuntimeError('Do not describe unvalidated arithmetic as complete')
    agg=read_csv(paper/'aggregate_metrics.csv');runtime=read_csv(paper/'runtime_metrics.csv')
    def get(condition,metric,backend='',group='',aggregation='micro'):
        return next(r for r in agg if r['condition']==condition and r['metric']==metric and r['aggregation']==aggregation and r.get('backend','')==backend and r.get('group','')==group)
    def percent(v):return 'N/A' if number(v) is None else f'{100*float(v):.3f}%'
    conditions=['full','no_identity_lock','no_blur_fallback','selective_blur_only','no_mouth_paste','no_swap_cache']
    smoke=[]
    for name in ['smoke_v1','smoke_v2','smoke_v3','smoke_v4']:
        path=run/name/'run_summary.json'
        if not path.exists():continue
        for row in json.loads(path.read_text()):
            smoke.append({k:row.get(k) for k in ['condition','clip_id','status','returncode','elapsed_sec']}|{'attempt':name})
    csv_write(paper/'smoke_run_status.csv',smoke)
    historical={}
    for condition in conditions[:3]:
        path=ROOT/'paper_results/panoptic20_ablation'/condition
        rows=read_csv(path/'per_clip_metrics.csv');stored=json.loads((path/'aggregate_metrics.json').read_text())
        non=sum(int(r['non_target_face_rows']) for r in rows)
        swapped=sum(int(r['non_target_swap_rows']) for r in rows);blurred=sum(int(r['non_target_blur_rows']) for r in rows)
        coverage=sum(float(r['target_coverage']) for r in rows)/len(rows)
        actual={'clips':len(rows),'clip_macro_system_target_coverage':coverage,'nontarget_face_rows':non,
            'swapped':swapped,'blurred':blurred,'micro_swap_coverage':swapped/non if non else None,
            'micro_blur_coverage':blurred/non if non else None,
            'stored_mean_target_coverage':stored['mean_target_coverage'],
            'recomputed_matches_stored_rounded':abs(coverage-stored['mean_target_coverage'])<1e-6}
        if not actual['recomputed_matches_stored_rounded']:raise RuntimeError('Historical target coverage failed independent recalculation')
        historical[condition]=actual
    write_json(paper/'historical_recalculation.json',historical)
    condition_manifest=json.loads((paper/'condition_manifest.json').read_text())
    path=paper/'dataset_manifest.jsonl'
    public_manifest=[json.loads(s) for s in path.read_text().splitlines()]
    for row in public_manifest:
        if 'target_manually_replaced' in row:
            row['historical_target_manually_replaced']=row.pop('target_manually_replaced')
        if row.get('reinstated') and 'target_face_index' in row:
            row['historical_neighbor_seed_face_index']=row.pop('target_face_index')
            row['reference_index_note']='Seed index belongs to the neighboring historical frame; selected_reference_bbox is the actual independently detected new reference'
    path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in public_manifest))
    condition_manifest['implementation_map']={
        'full':{'functions':['prepare_track','FaceSwapper.swap_many_into_frame','paste_cached_swap','finalize_track']},
        'no_identity_lock':{'config':'IDENTITY_LOCK_ENABLED=False','function':'prepare_track; skip temporal lock branch','retained':['get_track_embedding cache','BoT-SORT','stable-ID association']},
        'no_blur_fallback':{'config':'ENABLE_FALLBACK_BLUR=False','function':'finalize_track; failed/ineligible uncached faces remain UNPROCESSED'},
        'selective_blur_only':{'config':'ENABLE_FACE_SWAP=False','function':'main skips swapper and cache read; finalize_track blurs background'},
        'no_mouth_paste':{'config':'ENABLE_MOUTH_PASTE=False before import','function':'FaceSwapper original-mouth compositing guard'},
        'no_swap_cache':{'config':'ENABLE_SWAP_CACHE=False','function':'main disables swap_patch_cache writes and paste_cached_swap reads'}}
    condition_manifest['declared_but_unused_config_keys']=['STITCH_BLUR_KERNEL','SIDE_FACE_CENTER_RATIO_THRESHOLD','ENABLE_OCCLUSION_FALLBACK','ENABLE_MASK_BLEND','SWAP_FEATHER_RATIO','SWAP_MASK_BLUR_KERNEL']
    condition_manifest['common_auxiliary_state']={'embedding_cache_max_age_frames':30,'embedding_refresh_interval':5,'target_hold_frames':60,'cache_spatial_jump_ratio':.75}
    write_json(paper/'condition_manifest.json',condition_manifest)
    now=datetime.datetime.now(datetime.timezone.utc);job=json.loads((run/'primary_job.json').read_text())
    started=datetime.datetime.fromisoformat(job['started_utc']);finished=datetime.datetime.fromtimestamp((run/'primary/benchmark_summary.json').stat().st_mtime,datetime.timezone.utc)
    stages=json.loads((run/'postprocessing_status.json').read_text())['stages']
    timing={'primary_processing_started_utc':started.isoformat(),'primary_driver_finished_utc':finished.isoformat(),
        'primary_driver_wall_sec':(finished-started).total_seconds(),
        'primary_subprocess_wall_sec_sum':sum(float(r['end_to_end_sec']) for r in runtime),
        'postprocessing_active_stage_sec_sum':sum(s['elapsed_sec'] for s in stages.values()),
        'elapsed_from_primary_start_sec':(now-started).total_seconds(),
        'note':'Primary driver includes hashing/output validation; subprocess FPS excludes those. Model/data preparation preceded this start. Waiting controller intervals are not double-counted.'}
    branch=subprocess.check_output(['git','branch','--show-current'],text=True,cwd=ROOT).strip()
    reflog=ROOT/'.git/logs/refs/heads'/branch
    if reflog.exists():
        match=re.search(r' (\d+) [+-]\d{4}\t',reflog.read_text().splitlines()[0])
        if match:
            created=datetime.datetime.fromtimestamp(int(match.group(1)),datetime.timezone.utc)
            timing.update(branch_created_utc=created.isoformat(),elapsed_since_branch_creation_sec=(now-created).total_seconds(),
                branch_elapsed_note='Includes the interrupted/idle interval between user turns; not GPU compute time')
    write_json(paper/'timing_summary.json',timing)
    lines=['# VEIL 실행 결과 안내', '',
        '주 평가 21개 클립 ×6조건(126회)을 완료했습니다. 별도 ChokePoint 2개 촬영 ×6조건은 12회 시도하여 6회 완료·6회 등록 실패했고, 실패를 분모에 남겼습니다. '
        '독립 AdaFace·FaceNet 평가, 같은 코덱 대조군, 집계·paired 비교·cluster bootstrap 및 PNG/SVG 그래프를 생성했습니다. '
        '**수치 무결성 검증과 사람 주석 검증은 별개입니다. 사람 검증은 대기 상태이며 결과는 탐색적입니다.**', '',
        '처리 코드: `'+job['processing_code_commit']+'`. 평가 코드는 provenance에 따로 기록했습니다. '
        '과거 Panoptic-20 결과는 원본 그대로 보존했습니다.', '',
        '## 주 평가 결과', '',
        '아래 비율은 미검증 공식 3D 투영 주석 중 가시성 적격 얼굴 관측의 micro 집계입니다. '
        '잠재 노출은 preserve/unprocessed/missed/unknown을 포함하며 실제 재식별률과 다릅니다.', '',
        '| 조건 | 대상 preserve | 비대상 잠재 노출 | 비대상 swap | 비대상 blur | E2E FPS |',
        '|---|---:|---:|---:|---:|---:|']
    for c in conditions:
        vals=[percent(get(c,m)['value']) for m in ['derived_target_preserve_rate','derived_nontarget_exposure_rate','derived_nontarget_swap_fraction','derived_nontarget_blur_fraction']]
        fps=float(get(c,'end_to_end_fps',aggregation='pooled_frames_over_wall_time')['value'])
        lines.append('| '+c+' | '+' | '.join(vals)+f' | {fps:.2f} |')
    lines+=['','## 독립 재식별 검증','','성공적으로 점수가 계산된 비대상 관측의 수락률입니다. '
        '검출·정렬·임베딩 실패를 성공으로 세지 않았으며, 전체 시도에 대한 하한/상한을 함께 표시합니다. '
        '매 5프레임을 평가했고, 임계값은 처리 영상 평가 전에 별도 LFW에서 고정했습니다.','',
        '| 조건 | verifier | 수락/점수 계산 | 전체 시도 | 조건부 수락률 | 전체 시도 위험 범위 |',
        '|---|---|---:|---:|---:|---:|']
    for c in conditions:
        for b in ['adaface','facenet']:
            r=get(c,'accept_rate_successful_pairs',b,'non_target');lo=get(c,'accept_rate_lower_bound_all_attempts',b,'non_target');hi=get(c,'accept_rate_upper_bound_all_attempts',b,'non_target')
            lines.append(f'| {c} | {b} | {int(float(r["numerator"]))}/{int(float(r["denominator"]))} | {int(float(lo["denominator"]))} | {percent(r["value"])} | {percent(lo["value"])}–{percent(hi["value"])} |')
    lines+=['','LFW 보정은 100명, genuine 100쌍, impostor 9900쌍의 경험적 FMR 1% 기준점입니다. '
        '쌍 사이 의존성과 영상 도메인 차이 때문에 낮은 모집단 FMR 보장을 주장하지 않습니다. '
        '원본 인식률, 대상 유지율, 원본 검증 가능 subset, swap/blur 분리, person/clip any-frame 및 평균 임베딩 공격은 '
        '`independent_identity_verifier/`와 집계 CSV에 있습니다. 다중 프레임 공격의 FMR은 별도 보정되지 않았습니다.','',
        '## 파일과 재현','','- `dataset_manifest.jsonl`, `model_registry.json`, `per_run_provenance.jsonl`: 입력·모델·코드·출력 해시.',
        '- `per_clip_metrics.csv`, `per_identity_metrics.csv`, `runtime_metrics.csv`, `quality_metrics.csv`: 실제 측정값.',
        '- `aggregate_metrics.*`, `paired_comparisons.csv`, `confidence_intervals.csv`, `leave_one_sequence_out.csv`: 재계산 가능한 통계.',
        '- `integrity_checks.json`, `raw_validation.json`, `validation_checks.csv`: 실행 무결성과 독립 raw 재계산.',
        '- `figures/`: 7개 주 그래프의 PNG/SVG 및 관련 원자료.',
        '- `chokepoint_supplement/`: 서로 다른 촬영 2개의 분리된 탐색적 결과.',
        '- `historical_stress_manifest.jsonl`: 복구한 제외 후보 9개. 원래 대상 신원/참조 소실로 처리는 대기.',
        '- `completion_status.json`: 실행 완료·검증 수준·미생성 항목의 이유.',
        '- [전체 재현 명령](../../docs/REPRODUCE_RESULTS.md), [지표 정의](../../docs/METRIC_DEFINITIONS.md), [제한사항](../../docs/EXPERIMENT_LIMITATIONS.md).','',
        '```bash',f'.venv/bin/python -m tools.aggregate_research_results --output-dir paper_results/{run.name}',
        f'.venv/bin/python -m tools.plot_research_results --output-dir paper_results/{run.name}','```','',
        '집계·그래프 재생성은 커밋된 경량 CSV만으로 가능합니다. 프레임 재평가에는 적법하게 확보한 원본·모델이 필요합니다. '
        '원본·처리 영상, 얼굴 crop, 임베딩, 대형 per-frame 자료는 Git에서 제외했습니다.','',
        '## 아직 검증되지 않은 항목','','사람의 대상 신원·투영·가림·짧은 오류 사건 검토가 필요합니다. '
        '로컬 `annotation_review.html`은 H264 원본 프록시와 수정 가능한 주석을 제공합니다. '
        '확인한 관측만 human_verified로 기록하며 모델 보조 주석을 사람 GT로 부르지 않습니다. '
        '과거 수동 교체 참조 4개의 실제 이미지와 원래 교체 얼굴 해시는 없어 동일 입력 재현을 주장할 수 없습니다.','',
        '기존 0% 노출/100% 처리는 시스템 라벨 지표입니다. 이번 독립 평가와 분모·판정 근거가 다릅니다. '
        '새 수치를 옛 수치에 맞추지 않았으며, 모델·입력 재구성과 정확성 수정의 영향을 한 가지 원인으로 분리할 수 없습니다. '
        '완전한 익명성, human MOS, 검증된 pose/expression 품질 또는 실시간 배포 성능을 주장하지 않습니다.','',
        f'주 처리 드라이버 {timing["primary_driver_wall_sec"]/60:.1f}분, 처리 시작부터 후처리·패키징까지 {timing["elapsed_from_primary_start_sec"]/60:.1f}분. '
        '환경은 RTX PRO 6000 Blackwell 96GB, Python 3.12, torch 2.10.0+cu128, ORT GPU 1.23.2입니다. '
        '자세한 시간 범위와 중단 시간은 timing_summary.json, 실제 provider/메모리는 runtime_metrics.csv를 확인하십시오.','']
    (paper/'README.md').write_text('\n'.join(lines))
    completion['historical_stress_processing'].update(recovered_intervals=9,decoded_intervals=9)
    completion['smoke_attempts']={'completed':sum(r['status']=='completed' for r in smoke),'failed':sum(r['status']!='completed' for r in smoke),'failed_setup_reason':'Interrupted YOLO checkpoint before any decoded processing'}
    write_json(paper/'completion_status.json',completion)
    print('Wrote actual-result guide, historical recalculation and delivery timing receipts.')


if __name__=='__main__':main()
