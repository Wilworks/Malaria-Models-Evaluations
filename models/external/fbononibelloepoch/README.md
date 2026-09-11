---
language: en
license: apache-2.0
library_name: ultralytics
tags:
  - yolov8
  - object-detection
  - computer-vision
  - medical-imaging
pipeline_tag: object-detection
inference: true
---

# Malaria Detection YOLOv8 Model

This model detects malaria parasites in blood smear images using YOLOv8.

## Model Details
- Type: Object Detection
- Architecture: YOLOv8
- Task: Malaria Parasite Detection

## Usage
"""
#python

from ultralytics import YOLO
#Load model
model = YOLO('fbononibelloepoch/malaria-detection')
#Perform inference
results = model('image.jpg')
"""
