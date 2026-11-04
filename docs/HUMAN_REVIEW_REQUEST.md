# Bundled human review request

No human verification has been claimed for this run. The implementation,
processing, real-model verification and numerical audits proceed independently.
The following input is needed for stronger annotation/identity claims:

1. Check all 21 reconstructed target references against the unprocessed clips and
   nearest official person-ID mappings. Pay particular attention to the four
   historical manual replacements in DATASET_AND_ANNOTATIONS.md and the reinstated
   135–145 s interval. If original manually replaced images still exist elsewhere,
   provide their source/generation information and hashes; current recorded crops
   cannot establish that historical equivalence.
2. Review projected boxes, frame alignment, camera visibility and occlusion.
   Inspect every short target-alteration/potential-exposure event with neighboring
   frames; do not infer absence from a detector failure. Source labels cover every
   decoded frame where official 3D faces are available. Unverified/ineligible
   observations must retain their status. The local overlay supports per-frame
   edits, missing-face additions and explicit target mapping confirmation.
3. P2L_S5_C2: its frozen largest-face reference cannot be registered by ArcFace.
   Confirm the intended same person and supply a usable same-person reference
   with provenance. All six failures stay in the current results. A changed
   registration requires all six conditions for that clip in a new run.
4. Nine rejected Panoptic intervals were recovered and decoded. Review the two
   historically blipping cameras separately from the seven reference-pose/quality
   cases. Original protected identities/rejected crop bytes are absent. Use the
   private stress review page to establish target/reference provenance before
   claiming a faithful target-aware stress experiment.

Run the local server/annotation commands in REPRODUCE_RESULTS.md. Return one
editable correction file with an actual reviewer ID/timestamp and any reference
provenance notes. A mapping confirmation verifies identity only; it does not
automatically verify all projected boxes. A partial review is not full human GT.

Private paths relative to the repository:

- `local_research/veil_eval_20261104_0200/annotation_review.html`
- `local_research/veil_eval_20261104_0200/stress/stress_review.html`
- `local_research/veil_eval_20261104_0200/qualitative_candidates/` (generated from
  measured score/failure rules after scoring; publication rights are not approved)

Do not send credentials or release face imagery through Git. Source, derivative
videos, crops and embeddings remain local; corrected annotation results should be
written to a new evaluation directory with their validation scope recorded.
