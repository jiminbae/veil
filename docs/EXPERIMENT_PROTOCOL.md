# VEIL evaluation protocol v1 — frozen 2026-11-04 (UTC+09)

Run ID: `veil_eval_20261104_0200`. Baseline: `e14d89f82fe7610965513996e686739046fd90e1`.
This is an executable research protocol and results guide, not a manuscript.
No new evaluation outputs were inspected before this protocol was written.

## Scope and population

Preserve one registered target face and selectively process other faces. Evaluate
face-based verification only; bodies, clothing, voices and background are outside
scope. Existing Panoptic-20 is historical/development evidence, already inspected
by its original authors. It is not an unseen generalization set.

Primary intended population: all 20 historical intervals plus
`170221_haggling_b3_hd_00_27_0001350000`, excluded historically only to obtain a
multiple of ten. All intervals are ten seconds, original HD resolution and native
frame rate. Do not exclude valid difficult clips for model failure. Corrupt or
unobtainable inputs remain in the denominator/status ledger. No scene duplication
to achieve a clip count. Historical stress candidates are reconstructed only
when sufficient interval provenance exists.

Reconstruct historical references using the manifest timestamp and integer crop
box. Historical image hashes are unavailable, including four manual replacements;
these references are reconstructions, not verified identical historical assets.
For the reinstated clip, select the independently detected face nearest the
previous interval's recorded reference center at the midpoint. Record this rule
and require human review; never change a reference after viewing condition scores.
Use the repository's existing synthetic replacement image with a recorded hash.

## Conditions and correctness changes

Required: full, no_identity_lock, no_blur_fallback, selective_blur_only.
Additional: no_mouth_paste, no_swap_cache (the patch cache is read by the render
path and therefore applicable). All use identical target registration, detection,
tracking, input resolution, thresholds and codecs. no_identity_lock retains the
embedding cache, tracking and stable-ID association; it is not frame-independent.
All benchmark outputs must be clean. Optional annotated video is separate.

Before running conditions, repair demonstrated bookkeeping/rendering defects:
no-op blur success, target overlap corruption, self-renewing locks, stale target
face membership, omitted empty frames, unsafe result reuse and destructive reruns.
These change the implementation relative to historical results. Conservative
expiry of locks is a behavioral correction; report its possible utility cost.
No threshold or face-quality tuning based on test results is permitted.

Run one fixed first-lexicographic clip across all six conditions as smoke test.
If correctness fails, record the reason and re-run affected smoke conditions in
a new directory. Full runs use the frozen code commit. One worker initially,
batch size 8 (swap stage); increase concurrency only based on resource measurement.

## Annotations and matching

Investigate official person IDs, face landmarks, HD synchronization and camera
calibration. Projected official 3D landmarks, if obtainable, remain derived labels
until projection, visibility and occlusion are verified. Otherwise use an
independent MTCNN detector and explicitly unverified identity association as
exploratory pseudo-labels. Never copy VEIL target/stable IDs into ground truth.
Persist clip_id, one-based frame_idx, person_id, is_target, bbox, visibility,
annotation_source and verification_status. Human-reviewed GT requires an actual
human review; AI inspection is not human verification. Provide an editable local
review tool and a single bundled review request.

Match within each frame one-to-one at IoU >= 0.5. Record unmatched faces and
duplicate detections. Every decoded frame appears in a frame ledger, including
empty frames. An absent observation alone cannot establish that a person exited.
Track events on consecutive annotated frames and disclose annotation stride;
never claim sub-stride event recall from sampled annotations.

## Independent verifiers and calibration

Prefer official AdaFace R50 and FaceNet InceptionResnetV1 VGGFace2. No VEIL
buffalo_l recognizer/cache reuse. AdaFace: five-landmark similarity alignment,
112x112 BGR, (pixel/255 - .5)/.5. FaceNet: official MTCNN crop at 160x160 RGB,
(pixel-127.5)/128. Both normalized embeddings and cosine similarity. Store exact
checkpoint/source-code hashes and actual detector/provider details.

Choose calibration data with identities/sequences disjoint from the Panoptic
evaluation and freeze thresholds before processed-video scoring. Select the
smallest representable cosine threshold above the empirical upper impostor
quantile for requested FMR=1%, and report genuine acceptance plus exact binomial
uncertainty and dependence limitations. If disjoint labeled calibration is not
available, report similarity scores only; thresholded acceptance is unavailable.
Do not claim very low FMR from a small/dependent impostor sample.

Each attempted face remains in the score ledger. Report source and output
detection/alignment/embedding success separately. Failures are unknown, never
anonymization success. Target: reference-to-original and reference-to-output.
Non-target: original-to-output, separated into SWAP/BLUR/other. Report all attempts
and original-verifiable subsets. Person/clip any-frame acceptance and mean
normalized embedding attacks use only independently associated identities and
report the number of sampled frames. Same-frame source comparison is a threat
model, not an independent enrollment photograph.

## Measures and statistics

Keep system-log, pseudo-label/GT, verification, utility and runtime measures
separate. Primary intended endpoints: target preservation and non-target original
identity acceptance. They cannot become validated GT endpoints without validated
annotations/calibration. Missing measures are null with a reason, never zero.
Preserve, swap, blur, unprocessed, unknown and missed counts have explicit
denominators. Report per-clip, micro totals and unweighted clip-macro rates.

Paired comparisons use identical clips. Cluster bootstrap by source sequence,
seed 20261104, 2000 resamples, percentile 95% intervals; also report number of
clusters, worst clips and leave-one-sequence-out sensitivity. Continuous frames
are not independent units. Repeated cross-camera people may span sequences:
sequence clustering does not establish person independence.

Utility: independent detector retention and target ROI change against a same-codec
unprocessed control. Pixel distance is not face-swap perceptual quality. No MOS,
expression accuracy or pose accuracy without the required independent evidence.
No unaligned temporal pixel differences interpreted as flicker. Report action
switches with observed transition denominators.

Runtime: subprocess wall time includes startup/model load/encoding; distinguish
model initialization, warm frame stages and encoder draining. Synchronize CUDA
when measuring GPU stages. Record actual ONNX session providers, resolution,
face counts and peak memory. Do not infer real-time capability from enqueue time.

## Integrity, restart and amendments

Reuse only if code/config/input/reference/replacement/model and output hashes
match a completed receipt. Never overwrite a completed or failed attempt silently.
No original videos, face crops, processed videos, third-party weights or per-frame
biometric records in Git. Panoptic forbids redistribution of modified data without
permission. Lightweight aggregates and statistical figures only.

Record all attempts, failures and restart commands. An implementation fix after
the code freeze requires an amendment and consistent rerun of affected conditions.
Rebuild aggregate tables and figures from actual per-clip rows; assert unique
condition/clip keys, denominators, input hashes and frame counts. Completion is
separate for implementation, execution and validation.

## Amendment log

- v1: initial freeze; source reconstruction and human verification are pending.
- v1.1 (before test scoring): official `hdFace3d` per-frame JSON is accessible.
  Projection uses camera R/t/K/distortion, positive depth, >=12 landmarks with
  averageScore >=0.1 and averageReproError <=10. Expand landmark bounds by 15%
  left/right, 35% top and 5% bottom. Camera-visible eligibility requires >=5
  landmarks whose official visibility list contains the HD node. Preserve rows
  failing visibility eligibility in the annotation ledger. Source HD index is
  ceil(start_seconds * decoded native fps) + local one-based frame index - 1.
  Match the target at the reference timestamp by nearest projected face center
  to the recorded selected reference box. All projection and identity mappings
  require human verification. Per-frame system matching is exhaustive;
  recognition scoring samples every fifth frame (fixed before any test scores),
  and cannot establish every-frame re-identification or sub-stride events.
- Setup failures: an interrupted YOLO download caused all six smoke_v1 attempts
  to fail before decoding frames; full ZIP validation and a fresh download
  resolved it. AdaFace's official Lightning metadata requires an inert allowlist
  entry under torch weights_only; no checkpoint Python is executed.
- v1.2 (before primary runs): smoke_v2 completed six conditions on the first
  clip. Additional smoke inputs were selected only from original reference
  geometry to exercise eligible non-target swaps, not to improve metrics.
  smoke_v4 used the first interval with an in-bounds swap-sized non-target face:
  `170221_haggling_b2_hd_00_27_0001850000`. It confirmed 10 actual cache uses
  under full, 0 under no_swap_cache, and 0 swaps under selective_blur_only.
  Primary conditions remain unchanged and no clip is excluded for these findings.
