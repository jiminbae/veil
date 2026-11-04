# Data, reconstruction and annotation status

Primary intended population: Panoptic historical 20 plus the interval removed
solely to reach a multiple of ten. The 21st interval is
`170221_haggling_b3_hd_00_27_0001350000` (135–145 s). Selection, original source
URLs, timestamps, camera IDs and integer crop coordinates come from the retained
historical `accepted_manifest.jsonl` and `accepted_summary.json` at `e14d89f`.
No existing historical artifact is rewritten.

Clips are reconstructed at 1920×1080, native 30000/1001 FPS, 300 frames, ten
seconds, using accurate FFmpeg HTTP range seeking and MPEG-4 encoding. Each
decoded clip, PNG reference frame and PNG target crop has SHA-256. The full
original multi-minute video was not downloaded, so its full-file hash is null;
official URL, byte length, ETag and Last-Modified identify the ranged source.
HTTPS on the official Panoptic server failed certificate validation because of
its weak key; the original manifest's HTTP endpoint is used and disclosed.
File hashes provide local integrity, not authenticated HTTP-origin proof.
The requested interval is ten seconds, while 300 frames at native FPS span
10.01 seconds. The two adjacent intervals around the reinstated clip have a
one-frame boundary overlap under the explicit source-index map. They are not
independent clips; sequence clustering retains their dependence. Complete clip
hashes are unique. An exact-PTS source spot check at 135 s confirmed source frame
4046 as the first frame for `170221_haggling_b2_hd_00_27_0001350000`; other
timing/projection mappings remain unverified pending review.

The original input videos, original target crops, original replacement file and
per-frame benchmark outputs are absent from this checkout. Only historical
lightweight tables are available. No historical run is reused as a new run.
The tracked `media/fake_face.jpg` supplies the new replacement face; the original
historical replacement cannot be proven identical. Target registration and
replacement paths are distinct in the launcher and are hashed separately.

## Four manually changed historical references

The historical manifest flags these references as manually replaced:

| Clip | Source time (s) | Recorded crop x1,y1,x2,y2 |
|---|---|---|
| 170224_haggling_a2_hd_00_27_0001750000 | 180 | [1244,310,1454,613] |
| 170221_haggling_m1_hd_00_08_0001750000 | 180 | [258,179,474,517] |
| 170228_haggling_a1_hd_00_08_0002350000 | 240 | [1266,273,1472,535] |
| 170221_haggling_b1_hd_00_27_0001150000 | 120 | [697,479,946,839] |

Their actual historical bytes/hashes and manual-edit instructions are unavailable.
Regenerating the recorded crop therefore **cannot verify reproduction of the
manual replacement**. New references use the recorded source timestamp and crop,
are labeled reconstructed, and need human identity review. The reinstated clip
uses the independently detected face closest to the preceding interval's selected
reference center, as frozen before evaluation. References are never selected by
condition scores. The audit contains all 21 references, not just the four cases.

## Panoptic derived labels

Official source: [CMU Panoptic](http://domedb.perception.cs.cmu.edu/) and its
[toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox).
The official `hdFace3d/faceRecon3D_hd########.json` files contain person IDs,
70 three-dimensional landmarks, scores, reprojection errors and per-landmark
camera-visibility lists. Camera files supply R, t, K, distortion and dimensions.
The projection/box/visibility and temporal-index rules are in the frozen protocol.

Every requested frame's official JSON is retained locally. Derived rows include
clip_id, one-based frame_idx, official sequence-local person_id, target mapping,
bbox, visibility, annotation_source and verification_status. A nearest-reference
center mapping selects the official target ID. IDs do not establish global
identity continuity between recordings. Full-frame coverage and camera visibility
are counted separately. Derived labels are **unverified projections**, not human
GT; AI visual spot checks do not change that status.

The review page overlays labels on local unprocessed video, displays reconstructed
registration images, supports per-observation corrections and explicit target-ID
confirmation, and exports an editable correction JSON. Applying corrections writes
a new annotation directory. Only explicitly reviewed observations acquire human
verification; confirming an identity does not verify all boxes or frames.

## Secondary source and stress coverage

[ChokePoint](https://arma.sourceforge.net/chokepoint/) permits non-commercial
research under its NICTA license and requires attribution and derivative notices.
Required citation: Wong, Chen, Mau, Sanderson and Lovell, *Patch-based Probabilistic
Image Quality Assessment for Face Selection and Improved Video-based Face
Recognition*, CVPR Workshops 2011, pp. 81–88.

Use the two genuinely crowded recordings P2E_S5 and P2L_S5, camera C2 once each;
do not duplicate simultaneous views or adjacent windows to make ten examples.
Official files come from [Zenodo record 815657](https://zenodo.org/records/815657).
The crowded sequences have no identity/eye labels in the supplied groundtruth
archive. An independent MTCNN detector with geometry tracking creates explicitly
unverified pseudo-labels. Its target IDs are not human/official IDs, and sharing
MTCNN with the verifier can bias detector-retention results. Keep this small
secondary source separate from Panoptic aggregates.

Archived clips include side profiles, occlusions, small faces and the six known
historical target-alteration cases. These stay in the primary cohort. Historical
review CSVs are retained for difficult-case provenance. All nine rejected source
intervals were recovered from the retained summary IDs and decoded successfully.
Their manifest and browser review page are separate. Seven were rejected for
reference quality/pose and two for historical camera blipping. Successful decoding
does not prove the camera imagery is free of glitches. Original rejected crops and
protected identities were not retained, so faithful target-aware stress processing
is pending target registration/provenance review. No guessed target or fabricated
stress metric is supplied; this is a recovered-input companion, not a held-out
test. AVA was reviewed as an alternative, but its YouTube video IDs and person
boxes do not themselves establish video redistribution rights or face GT.

## Access and publication

Panoptic is research-only and forbids redistribution of the dataset or modified
data without organizer permission. All input/output videos, crops, screenshots,
per-frame labels, score records, embeddings and model checkpoints remain under
ignored `local_research/` or weights/cache directories. Git receives relative,
non-sensitive manifests, code, environment descriptions, aggregate numeric tables
and statistical plots. No face imagery or model weights is added by this work.
Existing tracked demonstration media is historical repository content.

Calibration uses the sklearn-distributed LFW archive, a distinct image corpus:
first 100 alphabetical identities with >=2 images, first two filenames each.
It is not an additional video-generalization experiment. The 100 genuine and
9900 cross-identity comparisons share images/identities. Empirical FMR is an
operating-point calibration, not a low-FMR population guarantee. No Panoptic or
ChokePoint image sets thresholds; a global subject registry is unavailable to
prove absolute cross-corpus person disjointness.
