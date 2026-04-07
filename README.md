# animalmodel02.ipynb - This file contain our base model
YOLOv8 Animal Detection Model

This repository contains a custom-trained YOLOv8 model for animal detection, developed as part of the Intelligent Animal Trap Utilizing Detection Technology project.

Model Overview
- Base Model: YOLOv8 Nano (yolov8n.pt)
- Framework: Ultralytics YOLOv8
- Task: Multi-class animal detection
- Training Platform: Google Colab
- Input Size: 640 × 640

Dataset
- Custom animal dataset
- Annotated in YOLO format
- Dataset configuration defined in data.yaml
- Stored in Google Drive and loaded during training

# MotionCheckwithModel.ipynb
Motion-Triggered YOLOv8 Inference

This script performs motion-based animal detection using a custom-trained YOLOv8 model. It first detects motion in a video stream, then runs YOLO inference only when motion is present to reduce unnecessary computation.
