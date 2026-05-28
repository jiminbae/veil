# DL-project

## Setup

```bash
git clone --recurse-submodules https://github.com/jiminbae/DL-project
cd DL-project
bash scripts/download_weights.sh
```

`download_weights.sh` downloads the required model files into
`models/project/face_tracking/weights/`:

- `yolo26x-face.pt`
- `inswapper_128.onnx`

Run the updated face-swapping pipeline with:

```bash
cd models/project/face_tracking
python main_hybrid.py
```
