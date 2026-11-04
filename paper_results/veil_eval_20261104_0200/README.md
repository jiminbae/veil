# VEIL 실행 결과 안내

주 평가 21개 클립 ×6조건(126회)을 완료했습니다. 별도 ChokePoint 2개 촬영 ×6조건은 12회 시도하여 6회 완료·6회 등록 실패했고, 실패를 분모에 남겼습니다. 독립 AdaFace·FaceNet 평가, 같은 코덱 대조군, 집계·paired 비교·cluster bootstrap 및 PNG/SVG 그래프를 생성했습니다. **수치 무결성 검증과 사람 주석 검증은 별개입니다. 사람 검증은 대기 상태이며 결과는 탐색적입니다.**

처리 코드: `df04db2ad5706dc27fe8a1d1f45aedb442d2800c`. 평가 코드는 provenance에 따로 기록했습니다. 과거 Panoptic-20 결과는 원본 그대로 보존했습니다.

## 주 평가 결과

아래 비율은 미검증 공식 3D 투영 주석 중 가시성 적격 얼굴 관측의 micro 집계입니다. 잠재 노출은 preserve/unprocessed/missed/unknown을 포함하며 실제 재식별률과 다릅니다.

| 조건 | 대상 preserve | 비대상 잠재 노출 | 비대상 swap | 비대상 blur | E2E FPS |
|---|---:|---:|---:|---:|---:|
| full | 97.438% | 4.636% | 23.422% | 71.942% | 10.92 |
| no_identity_lock | 91.152% | 4.623% | 23.422% | 71.955% | 10.83 |
| no_blur_fallback | 97.438% | 76.578% | 23.422% | 0.000% | 11.04 |
| selective_blur_only | 97.438% | 4.636% | 0.000% | 95.364% | 19.68 |
| no_mouth_paste | 97.438% | 4.636% | 23.422% | 71.942% | 11.17 |
| no_swap_cache | 97.438% | 4.636% | 19.421% | 75.943% | 11.15 |

## 독립 재식별 검증

성공적으로 점수가 계산된 비대상 관측의 수락률입니다. 검출·정렬·임베딩 실패를 성공으로 세지 않았으며, 전체 시도에 대한 하한/상한을 함께 표시합니다. 매 5프레임을 평가했고, 임계값은 처리 영상 평가 전에 별도 LFW에서 고정했습니다.

| 조건 | verifier | 수락/점수 계산 | 전체 시도 | 조건부 수락률 | 전체 시도 위험 범위 |
|---|---|---:|---:|---:|---:|
| full | adaface | 426/1086 | 1583 | 39.227% | 26.911%–58.307% |
| full | facenet | 99/1086 | 1583 | 9.116% | 6.254%–37.650% |
| no_identity_lock | adaface | 424/1084 | 1583 | 39.114% | 26.785%–58.307% |
| no_identity_lock | facenet | 97/1084 | 1583 | 8.948% | 6.128%–37.650% |
| no_blur_fallback | adaface | 1225/1484 | 1583 | 82.547% | 77.385%–83.639% |
| no_blur_fallback | facenet | 1143/1484 | 1583 | 77.022% | 72.205%–78.459% |
| selective_blur_only | adaface | 529/1034 | 1583 | 51.161% | 33.418%–68.099% |
| selective_blur_only | facenet | 80/1034 | 1583 | 7.737% | 5.054%–39.735% |
| no_mouth_paste | adaface | 369/1086 | 1583 | 33.978% | 23.310%–54.706% |
| no_mouth_paste | facenet | 82/1086 | 1583 | 7.551% | 5.180%–36.576% |
| no_swap_cache | adaface | 442/1069 | 1583 | 41.347% | 27.922%–60.392% |
| no_swap_cache | facenet | 99/1069 | 1583 | 9.261% | 6.254%–38.724% |

LFW 보정은 100명, genuine 100쌍, impostor 9900쌍의 경험적 FMR 1% 기준점입니다. 쌍 사이 의존성과 영상 도메인 차이 때문에 낮은 모집단 FMR 보장을 주장하지 않습니다. 원본 인식률, 대상 유지율, 원본 검증 가능 subset, swap/blur 분리, person/clip any-frame 및 평균 임베딩 공격은 `independent_identity_verifier/`와 집계 CSV에 있습니다. 다중 프레임 공격의 FMR은 별도 보정되지 않았습니다.

## 파일과 재현

- `dataset_manifest.jsonl`, `model_registry.json`, `per_run_provenance.jsonl`: 입력·모델·코드·출력 해시.
- `per_clip_metrics.csv`, `per_identity_metrics.csv`, `runtime_metrics.csv`, `quality_metrics.csv`: 실제 측정값.
- `independent_identity_verifier/extraction_stage_metrics.csv`: 원본·처리본 검출/정렬/임베딩 성공률과 미상 단계의 범위.
- `aggregate_metrics.*`, `paired_comparisons.csv`, `confidence_intervals.csv`, `leave_one_sequence_out.csv`: 재계산 가능한 통계.
- `integrity_checks.json`, `raw_validation.json`, `validation_checks.csv`: 실행 무결성과 독립 raw 재계산.
- `figures/`: 7개 주 그래프의 PNG/SVG 및 관련 원자료.
- `chokepoint_supplement/`: 서로 다른 촬영 2개의 분리된 탐색적 결과.
- `historical_stress_manifest.jsonl`: 복구한 제외 후보 9개. 원래 대상 신원/참조 소실로 처리는 대기.
- `completion_status.json`: 실행 완료·검증 수준·미생성 항목의 이유.
- [전체 재현 명령](../../docs/REPRODUCE_RESULTS.md), [지표 정의](../../docs/METRIC_DEFINITIONS.md), [제한사항](../../docs/EXPERIMENT_LIMITATIONS.md).

```bash
.venv/bin/python -m tools.aggregate_research_results --output-dir paper_results/veil_eval_20261104_0200
.venv/bin/python -m tools.plot_research_results --output-dir paper_results/veil_eval_20261104_0200
```

집계·그래프 재생성은 커밋된 경량 CSV만으로 가능합니다. 프레임 재평가에는 적법하게 확보한 원본·모델이 필요합니다. 원본·처리 영상, 얼굴 crop, 임베딩, 대형 per-frame 자료는 Git에서 제외했습니다.

## 아직 검증되지 않은 항목

사람의 대상 신원·투영·가림·짧은 오류 사건 검토가 필요합니다. 로컬 `annotation_review.html`은 H264 원본 프록시와 수정 가능한 주석을 제공합니다. 확인한 관측만 human_verified로 기록하며 모델 보조 주석을 사람 GT로 부르지 않습니다. 과거 수동 교체 참조 4개의 실제 이미지와 원래 교체 얼굴 해시는 없어 동일 입력 재현을 주장할 수 없습니다.

기존 0% 노출/100% 처리는 시스템 라벨 지표입니다. 이번 독립 평가와 분모·판정 근거가 다릅니다. 새 수치를 옛 수치에 맞추지 않았으며, 모델·입력 재구성과 정확성 수정의 영향을 한 가지 원인으로 분리할 수 없습니다. 완전한 익명성, human MOS, 검증된 pose/expression 품질 또는 실시간 배포 성능을 주장하지 않습니다.

주 처리 드라이버 55.9분, 처리 시작부터 후처리·패키징까지 90.8분. 환경은 RTX PRO 6000 Blackwell 96GB, Python 3.12, torch 2.10.0+cu128, ORT GPU 1.23.2입니다. 자세한 시간 범위와 중단 시간은 timing_summary.json, 실제 provider/메모리는 runtime_metrics.csv를 확인하십시오.
