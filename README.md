# BuildingYOLO

Property image classifier and detector using **EfficientNetB0** (99% accuracy) + **YOLOv8n** (bounding box detection).

## Live Demo
[HuggingFace Space](https://huggingface.co/spaces/balkotjokes/BuildingYOLO)

## Classes

| Class | Source |
|-------|--------|
| Exterior Facade | CMP Facade DB |
| Office Interior | MIT Indoor Scenes |
| Warehouse | MIT Indoor Scenes |
| Pipelines | Roboflow pipeline-tracks |

## Models

| Model | Accuracy | Task |
|-------|----------|------|
| EfficientNetB0 | 99% | Classification |
| YOLOv8n | ~65% mAP@50 | Detection + Bounding Boxes |

## Project Structure

```
src/
  data_ingestion.py      Download CMP Facade + MIT Indoor + Roboflow
  data_transformation.py tf.data pipeline + YOLO dataset builder
  model_trainer.py       ResNet50V2 (v1)
  resnet_trainer_v2.py   EfficientNetB0 (v2, 99% accuracy)
  yolo_trainer.py        YOLOv8n detection training + inference
  app.py                 FastAPI server with live analytics
static/
  index.html             Dark UI with bounding box overlay
logger.py                UTF-8 logging
exception.py             Custom exception handling
main.py                  Orchestrator
Dockerfile               HuggingFace Spaces compatible (port 7860)
```

## Setup

```powershell
# Install dependencies
.\install.bat

# Configure Roboflow API key (free account)
.\setup_roboflow.bat

# Train both models + serve
python main.py
```

## Commands

```powershell
python main.py                  # train both + serve
python main.py --train-only     # train only
python main.py --serve          # serve existing models
python main.py --resnet-only    # EfficientNetB0 only (~60 min)
python main.py --yolo-only      # YOLOv8n only (~90 min)
python main.py --skip-mit       # skip 2.4 GB MIT download
python main.py --show-graphs    # plot training graphs
```

## Hardware

Trained on AMD Ryzen 5 7535HS (CPU only):
- EfficientNetB0: ~60 min
- YOLOv8n: ~90 min
- Total: ~150 min

## Data Sources

- **CMP Facade DB** — Czech Technical University, direct HTTP
- **MIT Indoor Scenes** — MIT CSAIL, direct HTTP
- **Roboflow pipeline-tracks** — requires free API key from app.roboflow.com

## Web UI Features

- Bounding box overlay drawn on uploaded image
- Classification probability bars
- Live analytics dashboard (SSE, updates every 2s)
- Class distribution pie chart
- Prediction confidence history
