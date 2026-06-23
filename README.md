# LPR

License Plate Recognition (LPR) is a Python computer-vision project for detecting vehicles and license plates, rectifying plate crops, and reading plate characters with a custom OCR model.

The main pipeline combines:

- Ultralytics YOLOv8 for vehicle detection
- Hugging Face YOLOS for license plate detection
- WPOD-Net for license plate orientation and perspective correction
- A TPS + ResNet + BiLSTM + Attention OCR model for character recognition

## Repository Layout

```text
.
|-- pipeline.py                 # Main LPR_Pipeline class
|-- frame_prediction.py         # Single-frame demo from demo_footage/video3.mp4
|-- video_prediction.py         # Video demo that writes output.mp4
|-- data_utils.py               # Small data/file helpers
|-- demo_footage/               # Sample videos for local demos
|-- detection/yolos_model/      # YOLOS config and preprocessor files
|-- orientation/wpodnet/        # WPOD-Net orientation model code
|-- OCR/                        # OCR model used by the pipeline
|-- OCR_fine_tuning/            # OCR training, evaluation, demo, and LMDB tooling
`-- segmentation/               # Standalone plate segmentation project
```

## Main Pipeline

The reusable interface is `LPR_Pipeline` in `pipeline.py`.

```python
from pipeline import LPR_Pipeline

pipeline = LPR_Pipeline(ocr_weights_path="weights/OCR.pth")
```

Key methods:

- `predict_cars(image)`: detects car crops with YOLOv8.
- `predict_license_plate(image, for_affine=False, confidence_threshold=0.5)`: detects and crops a plate with YOLOS.
- `predict_affine_plate(image)`: rectifies a plate crop with WPOD-Net.
- `predict_plate_characters(image)`: runs OCR and returns the predicted plate text.

There is no web server or HTTP API in this repo; current usage is through Python scripts and training CLIs.

## Required Model Artifacts

Large model files are intentionally not committed. Before running the top-level demos, place these files in the expected locations:

```text
weights/
|-- OCR.pth
|-- wpodnet.pth
`-- yolov8m.pt

detection/yolos_model/
`-- model.safetensors
```

The YOLOS config in `detection/yolos_model/config.json` identifies the source model as `nickmuchi/yolos-small-finetuned-license-plate-detection`. The `.gitignore` excludes `weights/*` and `detection/yolos_model/model.safetensors`, so these artifacts must be downloaded, trained, or copied locally after cloning.

The segmentation demo also expects local files such as `segmentation/model_v2.pth` and `segmentation/plate.jpg` when using `segmentation/test.py`.

## Setup

There is no root `requirements.txt` or package manifest yet. Install dependencies in a Python environment appropriate for your PyTorch/CUDA setup.

Common dependencies inferred from the codebase:

```bash
pip install torch torchvision transformers ultralytics opencv-python pillow matplotlib numpy pandas tqdm pytesseract easyocr lmdb natsort nltk fire
```

For the segmentation subproject, a Conda environment file is provided:

```bash
conda env update -f segmentation/environment.yml
```

If you use `pytesseract`, install the system Tesseract binary separately for your OS.

## Run Demos

Run a single-frame prediction from `demo_footage/video3.mp4`:

```bash
python frame_prediction.py
```

Run video prediction from `demo_footage/video4.mp4` and write `output.mp4`:

```bash
python video_prediction.py
```

Both demo scripts use hard-coded sample paths and expect the required model artifacts listed above. They also open OpenCV preview windows.

## OCR Fine-Tuning

OCR training and evaluation live in `OCR_fine_tuning/`.

Train with the included LMDB-style dataset directories:

```bash
cd OCR_fine_tuning
python train.py \
  --train_data train_dataset \
  --valid_data test_dataset \
  --Transformation TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction Attn
```

Run OCR-only inference on an image folder:

```bash
cd OCR_fine_tuning
python demo.py \
  --image_folder <folder> \
  --saved_model <model.pth> \
  --Transformation TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction Attn
```

Run OCR evaluation:

```bash
cd OCR_fine_tuning
python test.py \
  --eval_data <lmdb_dataset> \
  --saved_model <model.pth> \
  --Transformation TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction Attn
```

Training outputs are written under `OCR_fine_tuning/saved_models/<exp_name>/`.

## Segmentation

The `segmentation/` directory is a standalone license plate segmentation project with its own README, model code, dataset loader, training loop, and environment file.

Expected dataset layout:

```text
segmentation/dataset/
|-- train/
|   |-- images/*.jpg
|   `-- masks/<image-name>.jpg.png
`-- val/
    |-- images/*.jpg
    `-- masks/<image-name>.jpg.png
```

Example training command:

```bash
cd segmentation
python train.py --dataset-dir ./dataset --output-dir . --epochs 30 --batch-size 2
```

Run segmentation evaluation through the training script:

```bash
cd segmentation
python train.py --test-only --resume <checkpoint.pth> --dataset-dir ./dataset
```

See `segmentation/README.md` for more detail and pretrained segmentation weight links.

## Tests

No pytest/unittest suite or CI configuration is currently present. Available checks are script-level demos and evaluation commands:

- `python frame_prediction.py`
- `python video_prediction.py`
- `OCR_fine_tuning/test.py`
- `segmentation/train.py --test-only ...`

## Current Caveats

- Model weights are required but not committed.
- Several scripts use hard-coded paths and are best treated as demos.
- No deployment workflow, Dockerfile, or packaged CLI is currently provided.
- Some data-generation helper scripts under `OCR_fine_tuning/` appear unfinished and may need cleanup before use.
