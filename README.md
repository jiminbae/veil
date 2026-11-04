# VEIL: Video Face Exchange with Identity Locking

VEIL is a target-aware video face-swap pipeline that preserves selected target identities and replaces non-target faces in video using face detection, tracking, recognition, and InSwapper.

## What VEIL Does

VEIL processes an input video frame by frame. It detects faces with YOLO, tracks them with BoT-SORT, identifies protected target people with InsightFace/ArcFace embeddings, and swaps only non-target faces with a configured virtual face. Target identities are left unchanged.

The pipeline also writes per-frame tracking metadata, face metadata, logs, and a clean output video. When a non-target face is too small, occluded, or otherwise unsuitable for swapping, VEIL falls back to blur instead of forcing a low-quality swap. Separate local review overlays support annotation checking.

## Demo

Source video: [Magnific](https://www.magnific.com/free-video/two-women-doing-makeup-tutorial_3696427#fromView=search&page=1&position=27&uuid=ed77d5e3-7d98-416e-95bc-fca3aee8556b)

Virtual face source: [This Person Does Not Exist](https://thispersondoesnotexist.com/)

### Original Input

![Original input video test.mp4](media/input_test.gif)

### Virtual Face

<img src="media/fake_face.jpg" alt="Virtual face image fake_face.jpg" width="220">

### VEIL Output

![VEIL main_hybrid output using test.mp4 and fake_face.jpg](media/output_target5.gif)

## Project Layout

```text
veil/
├── README.md
├── media/
│   ├── input_test.gif
│   ├── fake_face.jpg
│   └── output_target5.gif
├── scripts/
│   └── download_weights.sh
└── models/
    ├── requirements.txt
    ├── boxmot/
    │   └── models/
    └── veil/
        ├── main_hybrid.py
        ├── config.py
        ├── detector_tracker.py
        ├── face_identifier.py
        ├── face_inswapper.py
        ├── face_utils.py
        ├── metadata_manager.py
        ├── crop_manager.py
        ├── target/
        ├── virtual_face/
        ├── videos/
        ├── weights/
        └── outputs/

```

## Requirements

- Python 3.12 is the tested research environment.
- Tested core research pins are in `models/requirements-research.txt`; the historical environment inventory remains in `models/requirements.txt`.
- NVIDIA GPU + CUDA are recommended for practical runtime.
- CPU execution is possible by setting `device = "cpu"` in `models/veil/config.py`, but it will be significantly slower.
- `models/requirements-research.txt` pins the tested core packages, including ONNX Runtime GPU. VEIL records actual session providers and uses CUDA/CPU as available. TensorRT is optional.
- `curl` or `wget` is required for `scripts/download_weights.sh`.

## Setup

```bash
git clone https://github.com/jiminbae/veil
cd veil
python -m venv .venv
source .venv/bin/activate
pip install -r models/requirements-research.txt
bash scripts/download_weights.sh
```

`download_weights.sh` downloads the required model files into `models/veil/weights/`:

- `yolo26x-face.pt`
- `inswapper_128.onnx`

The BoT-SORT ReID weight is stored at `models/boxmot/models/osnet_x0_25_msmt17.pt`.

## Inputs

Place your files in these paths before running the pipeline:

- Input video: `models/veil/videos/test.mp4`
- Protected target images: `models/veil/target/target*`
- Replacement face image: `models/veil/virtual_face/fake_face.jpg`

Target images can be `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp`. You can provide multiple target images as long as their filenames match `target*`.

Use clear target images where the protected person's face is visible and easy to detect. The replacement face image should also contain one clear face, preferably front-facing or near front-facing.

You can change these paths and thresholds in `models/veil/config.py`.

## Run

```bash
cd models/veil
python main_hybrid.py
```

Each run automatically chooses the next output index by scanning the existing files under `outputs/`. For the tested CUDA build, independent verifiers and full research workflow, follow [the reproduction guide](docs/REPRODUCE_RESULTS.md).

## Outputs

VEIL writes generated files under `models/veil/outputs/`:

- Clean result video: `outputs/result/output_target{N}.mp4`
- Runtime log: `outputs/log/tracking_target{N}_log.txt`
- Face metadata: `outputs/metadata/face_metadata{N}.json`
- Tracking metadata: `outputs/metadata/tracking_metadata{N}.json`

The default result video has no face boxes or labels. Benchmark execution forces clean output. `DRAW_ANNOTATIONS` is an optional debug setting; debug videos must not be used for evaluation. The metadata files include frame numbers, raw track IDs, stable face IDs, bounding boxes, target match status, similarity scores, quality labels, fallback reasons, and actual rendering actions. Every decoded frame also has a frame/timing ledger.

## Configuration

The main settings live in `models/veil/config.py`:

- `VIDEO_PATH`: input video path.
- `TARGET_DIR` and `TARGET_PATTERN`: protected identity image gallery.
- `TARGET_IMAGE_PATH`: replacement face image used by InSwapper.
- `ENABLE_FACE_SWAP`: turn face swapping on or off.
- `MAX_SWAP_FACES_PER_FRAME`: cap the number of non-target faces swapped per frame.
- `SIM_THRESHOLD` and `TARGET_THRESHOLD`: identity matching thresholds.
- `FACE_SWAP_BATCH_SIZE`: batch size used by the swap stage.
- `USE_BOTSORT_REID`: enable or disable BoT-SORT ReID features.
- `DETECTION_CONF`: face detection confidence threshold.
- `SWAP_MIN_FACE_SIZE` and `SWAP_MAX_FACE_AREA_RATIO`: decide when a detected face is suitable for swapping.
- `ENABLE_MOUTH_PASTE`: keeps the original mouth area when blending a swapped face.

## Pipeline Summary

1. Detect faces using YOLO face weights.
2. Track face boxes across frames with BoT-SORT.
3. Extract face embeddings with InsightFace/ArcFace.
4. Match tracked faces against target images.
5. Preserve target identities.
6. Swap eligible non-target faces using `inswapper_128.onnx`.
7. Apply blur fallback when swap quality would be unreliable.
8. Save output video, logs, and metadata.

## Troubleshooting

- `No target images found`: add at least one image whose filename starts with `target` under `models/veil/target/`.
- `No valid target embeddings loaded`: the target image exists, but no face was detected. Try a clearer face image.
- `Cannot open video`: check that the input video exists at `models/veil/videos/test.mp4`, or update `VIDEO_PATH` in `config.py`.
- `Cannot read target image`: check `models/veil/virtual_face/fake_face.jpg`, or update `TARGET_IMAGE_PATH` in `config.py`.

## Notes

- GPU execution is enabled by default through `device = "cuda"` in `config.py`.
- ONNX Runtime uses the available provider stack, including TensorRT/CUDA when installed and available.
- Downloaded model weights, local input assets, outputs, logs, and metadata are ignored by git.
- Use VEIL only with appropriate consent and for lawful, responsible video processing.

## Research Artifacts

This repository also includes research artifacts for evaluating VEIL as a
selective face de-identification pipeline. Historical Panoptic-20 ablation
summaries are available under `paper_results/panoptic20_ablation/`, with protocol
and evaluation notes in `docs/`.

These historical artifacts are provided separately from the basic project demo.
The new evaluation protocol, independent AdaFace/FaceNet verification, annotation
review and reproduction commands are documented in [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md),
[docs/IMPLEMENTATION_AUDIT.md](docs/IMPLEMENTATION_AUDIT.md) and
[docs/REPRODUCE_RESULTS.md](docs/REPRODUCE_RESULTS.md). New run status and actual
evidence are stored under `paper_results/veil_eval_20261104_0200/`; human annotation
validation remains a separate status from execution.

## Developers

- [jiminbae](https://github.com/jiminbae)
- [TELLEGIT](https://github.com/TELLEGIT)
- [creatingweb123](https://github.com/creatingweb123)
- [jin23bsj](https://github.com/jin23bsj)

