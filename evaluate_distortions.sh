#!/bin/bash

# CBT Cognitive Distortions Dataset Evaluation
# 인지왜곡 (Burns' Cognitive Distortions) 데이터셋 평가

echo "CBT Cognitive Distortions Evaluation"
echo "Based on Burns' Cognitive Distortions"
echo "Target: Distortion pattern identification (all-or-nothing, overgeneralization, etc.)"
echo ""

# Run evaluation with distortions dataset
./evaluate_datasets.sh --dataset distortions "$@" 