# F1-Car-Detection-Yolov8-Vs-SSD

A computer vision project that detects and classifies Formula 1 cars by team from race broadcast footage, comparing **YOLOv8** and **SSD (VGG16)** object detection architectures across multiple training configurations.

**Course: ECE5303 — Computer Vision**
**Dataset:** [F1 Car Classification — Roboflow Universe](https://universe.roboflow.com/f1-car-detection-model/f1-car-classification)

---

## What It Does

Given F1 race video or images, the model draws bounding boxes around each car and labels it with its team:

| Team | Livery Color |
|------|-------------|
| Ferrari | Red |
| McLaren | Orange |
| Mercedes | Silver |
| Red Bull | Dark blue / yellow |

Both models were trained on 6,600+ labelled images and tested on real broadcast footage.

---

## Results Summary

| Metric | YOLOv8 (Best) | SSD (Best) | YOLOv8 Avg | SSD Avg |
|--------|--------------|------------|------------|---------|
| mAP50 | **0.9608** | 0.9241 | **0.9577** | 0.9117 |
| mAP50-95 | **0.7611** | 0.6608 | **0.7489** | 0.6572 |
| Recall | **0.9201** | 0.7457 | **0.9304** | 0.7461 |
| Inference Speed | **5.4ms/frame** | ~18ms/frame | — | — |

YOLOv8 outperforms SSD on every metric and runs **3× faster**. At 5.4ms per frame it comfortably meets real-time broadcast requirements (25fps = 40ms budget per frame).

---

## Project Structure

```
f1-car-detection-yolov8-vs-ssd/
├── YoloV8&Results.ipynb          # YOLOv8 training, evaluation, and video inference
├── SSD&Comparison.ipynb          # SSD training and cross-model comparison
├── f1_video.mp4                  # Raw input broadcast clip used for testing
├── f1_video_output.mp4           # Output clip with bounding boxes drawn
└── F1CarDetectionReport.pdf      # Full write-up: methodology, results, discussion
```

---

## How to Run

### Prerequisites

- Python 3.8+
- A GPU is strongly recommended (experiments were run on a Kaggle T4)

Install dependencies:

```bash
pip install ultralytics torch torchvision torchmetrics roboflow opencv-python
```

---

### YOLOv8 — Training & Inference

Open **`YoloV8&Results.ipynb`** in Kaggle or Jupyter and run all cells.

The notebook covers:
1. Dataset download from Roboflow
2. Fine-tuning `yolov8n` from COCO pretrained weights across 4 configurations (varying batch size and image resolution)
3. Evaluation (mAP50, mAP50-95, recall, precision)
4. Running inference on `f1_video.mp4` and saving the annotated output

> **Tip:** Use Kaggle over Google Colab — Kaggle's background execution keeps the session alive if you close the browser.

---

### SSD — Training & Comparison

Open **`SSD&Comparison.ipynb`** and run all cells.

The notebook covers:
1. Custom `Dataset` class to convert Roboflow labels to SSD's absolute-coordinate format
2. Fine-tuning `ssd300_vgg16` (torchvision) with a custom SGD training loop
3. Post-training metric computation using `torchmetrics`
4. Side-by-side comparison plots against YOLOv8

---

## Training Configurations Tested

Both models were tested across 4 configs varying batch size, image resolution, and learning rate:

| Config | Batch Size | Image Size | Learning Rate |
|--------|-----------|------------|--------------|
| A | 16 | 640 (YOLO) / 300 (SSD) | 0.01 |
| B | 8 | 640 (YOLO) / 300 (SSD) | 0.01 |
| C | 16 | 416 (YOLO) / 512 (SSD) | 0.01 |
| D | 16 | 640 (YOLO) / 300 (SSD) | 0.001 |

Best performing config for both models: **Config C**.

---

## Key Findings

- **YOLOv8 Config C** (416px input) achieved the best overall result at **mAP50 = 0.9608**
- Reducing resolution from 640→416 *improved* YOLOv8 accuracy, likely by forcing reliance on shape/colour over high-frequency texture
- SSD severely overfit — validation loss peaked around epoch 6–7 while training loss kept dropping, largely due to VGG16's 138M parameters being too large for the dataset size and a lack of augmentation in the training loop
- **Red Bull** was the easiest class for both models; **Mercedes** was the hardest (silver livery blends into backgrounds)

---

## Notes

- The dataset is available publicly on Roboflow Universe (link above). Download it directly inside the notebooks.
- `f1_video_output.mp4` is the pre-generated output — you can view it without running anything.
- For a full breakdown of the methodology, per-class results, and discussion see `F1CarDetectionReport.pdf`.
