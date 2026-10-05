#!/bin/bash
set -e

echo "================================================================="
echo "  WAM-Bench: Frontline YOLO Thick Smear Regional Adaptation Suite"
echo "  Started at: $(date)"
echo "================================================================="

cd /Users/wilfredayineasumboya/Desktop/Projects/bcm-projects/Malaria-Models-Evaluations

echo ""
echo ">>> [1/3] Running YOLO PELP (Parameter-Efficient Frozen Backbone)..."
./.venv/bin/python scripts/train_yolo_thick.py \
    --name yolo_thick_pelp \
    --freeze 10 \
    --epochs 15 \
    --batch 16 \
    --imgsz 640 \
    --lr0 0.001

echo ""
echo ">>> [2/3] Running YOLO Full End-to-End Retraining (Unfrozen)..."
./.venv/bin/python scripts/train_yolo_thick.py \
    --name yolo_thick_full \
    --epochs 15 \
    --batch 16 \
    --imgsz 640 \
    --lr0 0.0005

echo ""
echo ">>> [3/3] Ingesting Results into Table 5 & Recompiling Both Manuscripts..."
./.venv/bin/python scripts/update_table5_yolo.py

echo ""
echo "================================================================="
echo "  ALL YOLOV8 RUNS COMPLETE & BOTH MANUSCRIPTS FULLY UPDATED!"
echo "  Finished at: $(date)"
echo "================================================================="
