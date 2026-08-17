#!/bin/bash

# CBT Core Beliefs Dataset Evaluation

echo " CBT Core Beliefs Evaluation"
echo "Based on Beck's Cognitive Model"
echo "Target: Core belief identification (helpless, unlovable, worthless)"
echo ""

# Run evaluation with core_major dataset
./evaluate_datasets.sh --dataset core_major "$@" 