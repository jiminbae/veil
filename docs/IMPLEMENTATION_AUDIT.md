# Implementation and historical-artifact audit

Baseline is `e14d89f82fe7610965513996e686739046fd90e1`, clean main. Historical
tables are retained under `paper_results/panoptic20_ablation/` and hashed in the
new provenance receipt. None is relabeled as a corrected-code result.

| Item at baseline | Finding and disposition |
|---|---|
| Historical cohort | 20 clips ×3 conditions =60 runs; full/no_identity_lock/no_blur_fallback only |
| Historical full | mean_target_coverage 0.915706; swap 1957/9024=0.216866; blur 7067/9024=0.783134 |
| Historical no identity lock | mean_target_coverage 0.849342; direct system-label metric |
| Historical 0 exposure / 100% coverage | Derived from system target labels and logs; not independent identity verification or human GT |
| Historical visual review | 6 transient target-alteration clips, 8 clean candidates; review covers old full output, not this run |
| Reinstated candidate | Accepted 21st interval was excluded solely to obtain a multiple of ten; included in the new 21 |
| Target/replacement separation | Already implemented correctly by benchmark launcher at baseline; retained with regression coverage |
| selective_blur_only | Already implemented in runner, absent from old final table; executed in new matrix |
| Independent verifier | debug_color was a smoke test; AdaFace/MagFace/FaceNet/custom ONNX were placeholders. Actual AdaFace and FaceNet adapters now run; unused MagFace/custom ONNX remain explicitly unimplemented |
| GT evaluator | Existed but could reuse one system observation for multiple faces; fixed to one-to-one assignment; absence of GT now yields null rates |
| Original-mouth restoration | Real rendering path; no_mouth_paste disables it before module import. Actual smoke output hash differs from full |
| Swap patch cache | Real writes AND reads affect rendered output; not_applicable would be false. Swap-capable smoke shows 10 cache uses and 92 swaps under full vs0 cache uses and82 swaps under no_swap_cache |
| No-op blur logging | Previously unconditional blur_applied=True even for invalid/unchanged ROI; now reports actual pixel change and failure reason |
| Target overlaps | Background blur could overwrite target pixels; target raw/smoothed regions are restored after all composition. Non-target overlap exposure remains a disclosed policy tradeoff |
| Identity lock expiry | Previously renewed from its own locked state; stable target memberships did not expire. Renew only with fresh direct matches, expire stale face/track evidence and bind track hold to stable identity |
| Stale embedding cache | Could reuse failed-refresh embeddings indefinitely or across large track jumps; cache TTL and spatial-jump invalidation now prevent that reuse |
| Empty frames | Tracker was not aged without detections, and observation-derived totals omitted trailing empty frames. Tracker receives empty detection arrays and every decoded frame has a ledger entry |
| Clean output | Boxes/labels were drawn into evaluation output. Benchmark now forces clean output; local review overlays are separate browser canvas layers |
| Result reuse | Previously only output existence was checked. Completed receipts now require code/config/input/reference/replacement/model/output hashes |
| Destructive retries | Runner deleted previous paths; now refuses overwrite. Legacy Panoptic-10 wrapper defaults to unique run directories and no longer deletes old results |
| Partial weight download | Previous downloader trusted any nonempty file. Interrupted YOLO download reproduced this failure; pinned checksums, preservation of invalid files and atomic rename repair it |

Processing correctness changes were frozen in commit `df04db2`. Evaluation and
paired-statistics code have separate commits recorded in provenance. Lock/cache
expiry and target-priority composition change behavior, so historical metrics
cannot be attributed solely to one fix. No identity or quality threshold was tuned
against processed evaluation-video scores.

The corrected no_identity_lock condition still uses embeddings cached for up to
30 frames, stable-ID association and BoT-SORT tracking. It is not frame-independent.
Clean encoding uses OpenCV mp4v; H264 proxies are generated only for browser review
and never used for measurement. Per-stage synchronization, end-to-end wall time,
warm time and NVML process peaks have distinct scopes.

Tests reproduce pixel no-ops/overlaps, stale lock/cache expiry, track reuse,
one-to-one matching, retained verification failures, calibration ties, restricted
human-review scope, partial model downloads, micro/macro differences, cluster
units and unavailable-count rejection. Real two-model calibration and smoke
scoring supplement synthetic tests. The initial failed smoke and subsequent
successful runs stay in the execution evidence.
