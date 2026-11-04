# Metric definitions and denominators

All frame indices are one-based within a materialized clip. Reported faces are
face observations, not independent people. Missing rates are blank in CSV and
null in JSON. Counts of zero mean measured absence within the stated scope.

| Family | Measure | Numerator / denominator |
|---|---|---|
| System | preserve/swap/blur/unprocessed/unknown | final rendered action count / all system observations |
| System | target frame coverage | decoded frames with >=1 system-target observation / every decoded frame, including empty frames |
| System | fallback reason count | occurrences of each reason; reasons can co-occur and are not additive categories |
| Derived labels | target preservation | matched target faces with final action PRESERVE / all eligible projected target faces |
| Derived labels | target alteration | target SWAP + BLUR observations / all eligible projected target faces |
| Derived labels | target missed | unmatched eligible target faces / all eligible projected target faces |
| Derived labels | non-target false preservation | non-target faces matched to PRESERVE / all eligible projected non-target faces |
| Derived labels | potential exposure | non-target PRESERVE + UNPROCESSED + UNKNOWN + MISSED / all eligible projected non-target faces |
| Derived labels | detection recall | eligible faces matched at IoU >=0.5 / all eligible faces |
| Derived labels | swap/blur transition rate | adjacent-frame SWAP-to-BLUR or BLUR-to-SWAP transitions / adjacent pairs with both actions in {SWAP,BLUR} |
| Verification | original recognition baseline | reference-to-original accepted scores / successful baseline comparisons |
| Verification | target maintenance | registered-reference-to-output accepted / successful target comparisons |
| Verification | non-target re-identification | same-frame-original-to-output accepted / successful non-target comparisons |
| Verification | lower all-attempt bound | accepted / every attempted eligible sampled face |
| Verification | upper all-attempt bound | (accepted + unavailable comparisons) / every attempted eligible sampled face |
| Verification | source/output extraction success | respective successful detection+alignment+embedding attempts / all sampled eligible faces |
| Verification | original-verifiable subset | accepted output comparisons on original-accepted observations / scored output comparisons on that subset |
| Person/clip | any-frame risk | person/clip with >=1 accepted sampled frame / person/clips with >=1 score; separate upper bound includes unknown faces |
| Person/clip | pooled attack | cosine acceptance between normalized mean source/output embeddings; target uses registered reference instead of mean source |
| Utility | target ROI excess MAE | channel-weighted output-vs-input absolute error minus same-codec-control-vs-input error, divided by target ROI channel count |
| Runtime | end-to-end FPS | decoded input frames / subprocess wall seconds, including imports, model initialization and final encoder drain |
| Runtime | warm FPS | frames / main loop wall time, including decode and encoder drain |

`MISSED` is an absent matched system observation, not proof a face was untouched.
Potential exposure is conservative action accounting, not an independent
re-identification result. Registered target-region restoration takes priority
over overlapping non-target anonymization. A tiny pixel difference establishes
that rendering occurred, not that identity was hidden.

Panoptic labels are official-3D-derived projections with **no human validation**.
Rows lacking camera visibility eligibility are retained locally and counted as
ineligible; their processing quality and complete-occlusion behavior are unknown.
Annotation fetch failures differ from official face-annotation absence. No
system observation is copied into label identity or target status.

Matching is maximum cardinality then maximum total IoU, with each observation
used at most once within a frame. Duplicate stable IDs do not merge spatially
distinct observations. Stable-ID-to-person multiplicity is a diagnostic, not a
fully validated identity-switch error rate. Events are consecutive eligible
annotated frames; gaps end events and cannot establish real person exits.

Recognition samples frames 1,6,11,...,296. The main two-model evaluation uses
unrounded cosine scores and pre-frozen LFW empirical 1% FMR thresholds. Original
non-target baseline enrollment uses the first successfully extracted source
observation, excluding its identical source frame from baseline recognition.
The non-target attack has access to the original frame: stronger than a separate
enrollment-photo threat model. Missing source/reference/output embeddings remain
unknown. The old `eval_identity_verifier.py` CLI retains all attempted pairs but
its manually supplied threshold is explicitly uncalibrated; the matrix workflow
uses `score_research_identity.py` with calibration receipts.

Mean-embedding and any-frame attacks use the single-frame threshold; their FMR
is **not** separately calibrated. The per-identity files show frame counts and
missingness. These results must not be described as fixed-FMR multi-frame attack
comparisons. No unsampled-frame or video-wide anonymity guarantee follows.

Micro rates recompute summed numerators / summed denominators. Clip macro is the
mean of non-null clip rates; its valid clip count is explicit. Failed runs remain
in run_status and per_clip_metrics with unavailable rates. Sequence-cluster
bootstrap uses 2000 draws, seed 20261104, percentile 95% intervals. Paired deltas
are full minus comparison on identical clips. Recording-day bootstrap and
leave-one-sequence-out estimates expose sensitivity; three recording days do
not support strong population inference. People can recur across sequences.

Target MAE is not perceptual swap quality. Same-codec controls use OpenCV mp4v,
identical dimensions and FPS; changing neighboring regions can still change
compression residuals. Signed excess MAE may be negative. No MOS, expression
accuracy, unaligned flicker score, LPIPS or SSIM is invented.

NVML peak process memory is sampled at 0.25 seconds and covers both PyTorch and
ONNX allocations but may miss short peaks. PyTorch allocator peaks are reported
separately and omit ONNX memory. Model-load-after-import time excludes import-time
InsightFace initialization; end-to-end wall time includes it. Stage host timers
are synchronized after GPU stages. Writer enqueue time is not encoding time.
