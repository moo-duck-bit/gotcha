#!/bin/bash

# CBT Fine-grained Beliefs Dataset Evaluation
# 세부신념 (Detailed Cognitive Beliefs) 데이터셋 평가

echo "🔍 CBT Fine-grained Beliefs Evaluation"
echo "📚 Based on Beck's Cognitive Model"
echo "🎯 Target: Detailed belief identification (specific cognitive patterns)"
echo ""

# Run evaluation with core_fine dataset
./evaluate_datasets.sh --dataset core_fine "$@" 