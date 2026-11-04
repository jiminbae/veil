# Interpretation limits and outstanding validation

This work provides runnable code and actual measurements. It does not establish
complete anonymity or a publishable scientific claim without the qualifications
below. Execution and annotation validation have separate completion states.

1. **Human annotation review is pending.** Panoptic face boxes are unverified
   projections from official 3D faces and camera visibility lists. Target-ID
   selection, temporal alignment, occlusion and box padding need human checking.
   Eligible visibility can exclude precisely the difficult fully occluded faces.
   Counts of ineligible rows accompany results; they are not successes. Geometry
   tracking on the secondary source is a pseudo-label method, not official GT.
2. **Historical inputs cannot be exactly recovered.** Old video/crop/model hashes
   and original rendered outputs are absent. Four manual target replacements
   cannot be proven recoverable from the recorded crops. New PNG references and
   source clips are reconstruction inputs. The new replacement is the existing
   tracked synthetic demo face. Historical result files remain unchanged. Nine
   rejected source intervals were recovered and decoded; original target
   identities/rejected crops are absent, so faithful stress processing remains
   pending instead of silently registering a different easier person.
3. **Historical 20 is development evidence.** Its clips and known failures were
   previously inspected. Adding the arbitrary excluded interval repairs that
   exclusion but does not create an unseen test set. Differences from historical
   tables combine correctness changes, input reconstruction and environment
   changes; they do not isolate one algorithmic effect.
4. **Verification has a defined threat model.** AdaFace and FaceNet are independent
   of VEIL's buffalo_l recognizer. Non-target same-frame source/output comparison
   gives the attacker original pixels. It is not a conventional separate-photo
   enrollment test. Target references come from each clip's own midpoint; this
   is not a cross-session enrollment study. Every fifth frame is sampled;
   any-frame and pooled attacks
   only cover those observations. Body, clothes, voice and context are outside
   scope. No claim of zero face-based re-identification follows from system logs.
5. **Failed extraction is unknown.** Conditional successful-pair acceptance can
   look low when blurred faces are undetectable. All-attempt lower/upper bounds,
   source and output success counts, and original-verifiable subsets accompany
   it. Source-reference failures remain visible. Pooled and any-frame attacks
   reuse single-frame thresholds and do not have independently calibrated FMR.
6. **Calibration uncertainty and dependence.** The operating point is empirical
   FMR=1% on 100 LFW subjects, 100 genuine and 9900 dependent cross-identity pairs.
   Pair-level binomial intervals are descriptive, not independence guarantees.
   There is domain shift to video and no global participant registry proving
   cross-corpus person disjointness. No 1e-4/1e-5 FMR claim is supported.
7. **Few independent sources.** Primary sequences come from only three capture
   days and people may repeat across sequences. Sequence-cluster intervals and
   day-cluster sensitivity do not turn these into a large independent sample.
   The ChokePoint supplement contains only two distinct crowded recordings;
   duplicating camera views to obtain ten clips would overstate generalization.
8. **Target-priority overlap policy.** Restoring registered target boxes prevents
   another operation from changing those pixels but may preserve overlapping
   non-target pixels. A `SWAP` or `BLUR` action means some pixels changed, not
   that all face pixels or identity cues were removed. Unknown/missed labels
   contribute to potential exposure. Motion/occlusion can still defeat identity
   association. Expiring stale locks/caches can increase blur on difficult targets.
9. **Utility is narrowly measured.** Same-codec target ROI MAE isolates some
   compression error but is not perceptual quality, expression accuracy or pose
   accuracy. Video compression can couple nearby regions. Independent extraction
   retention mixes detection, alignment and recognition availability. No human
   MOS or evaluator agreement was fabricated; no unaligned temporal pixel score
   is interpreted as flicker.
10. **Environment is tested but outside FaceNet's advertised pins.** Official
    facenet-pytorch 2.6.0 pins torch<2.3 and torchvision<0.18, incompatible with a
    practical native Blackwell CUDA stack. The actual tested inference uses
    torch 2.10.0+cu128 / torchvision 0.25.0 and explicit no-deps FaceNet install.
    Its Pillow pin also differs in this environment. Tests, real calibration and
    actual scoring validate this run, not every unsupported configuration. A
    package metadata check is recorded separately from execution tests.
11. **Runtime scope.** Single-worker subprocess FPS includes model imports/load
    and encoder drain; warm FPS differs. NVML 0.25-second sampling may miss brief
    peaks. Stage time is synchronized host wall time, not a full GPU profiler.
    Download, integrity decoding and verification are excluded from processing
    FPS but included in research-workflow elapsed-time records. No real-time
    deployment claim is made.
    Conditions were run in a fixed order, once per clip, without randomized-order
    repeats or a hardware cold-cache reset. Per-clip distributions and paired
    differences do not isolate thermal/order/cache effects.
12. **Optional external baseline remains unavailable.** Official DeepPrivacy2
    code and checkpoint policy were inspected. Its unchanged setup pins SciPy
    1.7.1 and Pillow 8.3.1 plus an old Detectron2/DensePose revision. The exact
    SciPy binary dependency probe fails on the available Python 3.12; only that
    Python interpreter is present. PyTorch is built for CUDA 12.8, while nvcc is
    12.0. A separate legacy/ported environment and validation are required; no
    unexecuted external-method numbers are presented. Selective blur is the
    executed local baseline. Weights inherit their training-data licenses.

Panoptic raw/modified imagery is not redistributed. HTTP source transport and
checkpoint/data license metadata are recorded. Statistical figures contain only
aggregated numeric evidence. Local qualitative candidates require separate
publication-rights review before release.

To resolve pending human validation, run the local annotation review command in
REPRODUCE_RESULTS.md, inspect each requested identity/box/visibility/timing issue,
export corrections with an actual reviewer ID, then regenerate evaluation into a
new result run. Do not retroactively mark the current exploratory rows validated.
