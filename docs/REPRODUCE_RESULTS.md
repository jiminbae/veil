# Reproduce and resume the evaluation

Run commands from the repository root. Restricted media and embeddings stay
local. Use a fresh run ID/output directory for a changed configuration, code,
reference, model or annotation version. Existing attempts are preserved.

## Environment

The recorded run uses Python 3.12, torch 2.10.0+cu128, torchvision 0.25.0,
NumPy 1.26.4, SciPy 1.14.1, OpenCV 4.11.0.86, ONNX Runtime GPU 1.23.2,
InsightFace 0.7.3, Ultralytics 8.4.201 and BoxMOT 13.0.17. Exact package inventory
and hardware details are in the run's environment files. FFmpeg is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install numpy==1.26.4 scipy==1.14.1 pillow==13.0.0 \
  opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86 matplotlib==3.11.2 \
  scikit-image==0.26.0 scikit-learn==1.9.1 onnx==1.23.2 insightface==0.7.3 \
  ultralytics==8.4.201 boxmot==13.0.17 pytest==9.2.0 gdown==5.2.2 \
  requests tqdm nvidia-ml-py coloredlogs
.venv/bin/python -m pip uninstall -y onnxruntime
.venv/bin/python -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.23.2
.venv/bin/python -m pip install --no-deps facenet-pytorch==2.6.0
```

FaceNet's upstream package pins are older than the Blackwell-capable PyTorch
stack. The explicit no-deps install is deliberate; its inference path was tested
by real calibration/scoring. `pip check` metadata deviations are documented in
EXPERIMENT_LIMITATIONS.md. This workstation reused its existing CUDA PyTorch
read-only through a private local .pth; a new environment should install it as
above, without that machine-specific path. No global driver/package changes are
required. A clean-machine installation was not independently tested here.

On hosts where the agent sandbox hides GPU/network access, execute these commands
in a normal host terminal. Verify `torch.cuda.is_available()` and actual ONNX
session providers; provider availability is not proof of GPU execution.

## Models and source snapshots

```bash
bash scripts/download_weights.sh
mkdir -p local_research/vendor
curl -fL https://raw.githubusercontent.com/mk-minchul/AdaFace/master/net.py \
  -o local_research/vendor/adaface_net.py
.venv/bin/gdown 1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI \
  -O models/veil/weights/adaface_ir50_ms1mv2.ckpt
```

The model registry in the result directory gives the exact SHA-256 for every
downloaded model and architecture snapshot. Verify those hashes rather than
trusting moving `master`/release URLs. Never accept a partially downloaded model:
validate the ZIP archive or ONNX model before launching. The historical downloader
skips any nonempty file; the recorded setup detected and replaced an interrupted
YOLO download in a separate attempt. The corrected downloader now checks pinned
SHA-256, preserves invalid files and renames completed temporary downloads atomically.

VEIL initializes the official InsightFace buffalo_l package (research weights)
under its normal local model cache. Record every ONNX hash. FaceNet's official
VGGFace2 weights and MTCNN detector weights are obtained by facenet-pytorch.
The independent verifier never uses buffalo_l recognition embeddings or cache.

## Inputs, labels and frozen calibration

```bash
mkdir -p local_research/NEW_RUN/calibration
OMP_NUM_THREADS=2 .venv/bin/python -m tools.prepare_research_inputs \
  --output-dir local_research/NEW_RUN/inputs --workers 3
OMP_NUM_THREADS=2 .venv/bin/python -m tools.project_panoptic_annotations \
  --manifest local_research/NEW_RUN/inputs/dataset_manifest.jsonl \
  --output-dir local_research/NEW_RUN/annotations --workers 8
curl -fL https://ndownloader.figshare.com/files/5976018 \
  -o local_research/NEW_RUN/calibration/lfw.tgz
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m tools.calibrate_independent_verifiers \
  --lfw-archive local_research/NEW_RUN/calibration/lfw.tgz \
  --output-dir local_research/NEW_RUN/calibration/results \
  --adaface-model models/veil/weights/adaface_ir50_ms1mv2.ckpt
```

Freeze thresholds before scoring processed test videos. Existing calibration JSON
is immutable within a run. Calibration selection, checkpoint hash and exact pair
scores are retained locally. Small public summaries are packaged for Git.

## Actual processing

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m tools.run_veil_benchmark \
  --manifest local_research/NEW_RUN/inputs/dataset_manifest.jsonl \
  --output-dir local_research/NEW_RUN/primary \
  --replacement-image media/fake_face.jpg \
  --conditions full no_identity_lock no_blur_fallback selective_blur_only no_mouth_paste no_swap_cache \
  --python "$(pwd)/.venv/bin/python" --timeout-sec 900 --skip-existing
```

Use the frozen processing commit identified in provenance, and preserve its
`models/veil/*.py` plus benchmark-runner hashes. `--skip-existing` accepts only a
completed receipt with matching code/config/input/reference/replacement/model
and output hashes. A mismatched or failed attempt is never silently overwritten:
choose a new output directory and record the retry. Model/hash validation and
full output decoding are part of the runner. Run summaries are saved after each
clip so interrupted progress remains inspectable.

For the recorded workstation run, replace NEW_RUN with
`veil_eval_20261104_0200`. The running primary job's PID, start time, original
command and processing commit are in `local_research/.../primary_job.json`.
Do not launch a duplicate job while that PID is active. The postprocessing job
has its own stage-status ledger and log under the same private run directory.

## Evaluation and tables/figures

```bash
.venv/bin/python -m tools.evaluate_research_run \
  --manifest local_research/NEW_RUN/inputs/dataset_manifest.jsonl \
  --annotations local_research/NEW_RUN/annotations \
  --benchmark local_research/NEW_RUN/primary \
  --output-dir paper_results/NEW_RUN --local-output local_research/NEW_RUN/evaluation
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m tools.score_research_identity \
  --manifest local_research/NEW_RUN/inputs/dataset_manifest.jsonl \
  --annotations local_research/NEW_RUN/annotations \
  --benchmark local_research/NEW_RUN/primary \
  --calibration local_research/NEW_RUN/calibration/results \
  --output-dir paper_results/NEW_RUN/independent_identity_verifier \
  --local-output local_research/NEW_RUN/evaluation \
  --adaface-model models/veil/weights/adaface_ir50_ms1mv2.ckpt
.venv/bin/python -m tools.evaluate_research_utility \
  --manifest local_research/NEW_RUN/inputs/dataset_manifest.jsonl \
  --annotations local_research/NEW_RUN/annotations --benchmark local_research/NEW_RUN/primary \
  --controls local_research/NEW_RUN/evaluation/controls \
  --output paper_results/NEW_RUN/quality_metrics.csv
.venv/bin/python -m tools.aggregate_research_results --output-dir paper_results/NEW_RUN
.venv/bin/python -m tools.plot_research_results --output-dir paper_results/NEW_RUN
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q
```

Aggregate tables and PNG/SVG figures can be rebuilt from the committed per-clip
CSV files without the restricted source imagery. Frame-level re-evaluation needs
legally obtained source data and model weights. Raw pair scores/embeddings remain
local; lightweight per-clip and person/clip summaries are public.

## Human review and corrected annotations

```bash
.venv/bin/python -m tools.create_annotation_review --run-root local_research/NEW_RUN
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory local_research/NEW_RUN
```

Open `http://127.0.0.1:8765/annotation_review.html`. Review all target identities,
the four reconstructed manual references, 21st reference, alignment/visibility,
small/occluded faces and short failure events. Enter an actual reviewer ID and
export corrections. The one-time required human input is the reviewed correction
file; no model-assisted inspection is counted as human validation.

```bash
.venv/bin/python -m tools.create_annotation_review \
  --annotations local_research/NEW_RUN/annotations --corrections review_corrections.json \
  --output-dir local_research/NEW_RUN/annotations_reviewed
```

Use a fresh evaluation result directory with the reviewed annotations. Partial
review verifies only the explicitly reviewed observations. Preserve the original
unverified run and correction audit trail.
If review changes the registered target image or intended protected identity,
rerender all six conditions for the affected clips under a fresh run ID. Changing
only evaluation annotations requires evaluation/scoring reruns, not unchanged
video rendering. Never silently replace references in the recorded run.

## Secondary source

Download only one ChokePoint archive at a time, following its official server
request and NICTA license. Keep its license and derivative notice locally.

```bash
mkdir -p local_research/NEW_RUN/chokepoint
curl -fL https://zenodo.org/records/815657/files/P2E_S5.tar.xz -o local_research/NEW_RUN/chokepoint/P2E_S5.tar.xz
curl -fL https://zenodo.org/records/815657/files/P2L_S5.tar.xz -o local_research/NEW_RUN/chokepoint/P2L_S5.tar.xz
OMP_NUM_THREADS=2 .venv/bin/python -m tools.prepare_chokepoint_supplement \
  --archives local_research/NEW_RUN/chokepoint --output-dir local_research/NEW_RUN/supplement
```

Process its `supplement/inputs/dataset_manifest.jsonl` with the same six-condition
command into `supplement/benchmark`, then evaluate into a separate supplement
result directory. Its independent detector/geometry tracks are pseudo-labels.
Do not merge these two source recordings with the Panoptic population or relabel
their fragmented tracks as known people.
