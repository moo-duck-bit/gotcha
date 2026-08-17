#!/bin/bash

# CBT Dataset Evaluation Script
# 각 테스트 데이터셋에 대해 GOT 모델 성능 평가

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DATASET="all"
NUM_CASES=""

# Function to print usage
print_usage() {
    echo "CBT Dataset Evaluation Script"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset DATASET     Dataset to evaluate (core_major|core_fine|distortions|all) [default: all]"
    echo "  -n, --num-cases NUMBER    Number of cases to evaluate per dataset [default: all cases]"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                        # Evaluate all datasets with all cases"
    echo "  $0 -d core_major -n 10    # Evaluate only core beliefs with 10 cases"
    echo "  $0 -d distortions         # Evaluate only cognitive distortions"
}

# Function to print colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -n|--num-cases)
            NUM_CASES="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate dataset option
if [[ "$DATASET" != "all" && "$DATASET" != "core_major" && "$DATASET" != "core_fine" && "$DATASET" != "distortions" ]]; then
    print_colored $RED "Error: Invalid dataset option '$DATASET'"
    print_colored $YELLOW "Valid options: all, core_major, core_fine, distortions"
    exit 1
fi

# Check if we're in the correct directory
if [[ ! -f "cbt/cbt_main.py" ]]; then
    print_colored $RED "Error: cbt_main.py not found in cbt/ directory"
    print_colored $YELLOW "Please run this script from the project root directory"
    exit 1
fi

# Check if required datasets exist
print_colored $BLUE "🔍 Checking dataset files..."

if [[ "$DATASET" == "all" || "$DATASET" == "core_major" ]]; then
    if [[ ! -f "CBT-Bench/core/core_major_test.json" ]]; then
        print_colored $RED "❌ Core beliefs dataset not found: CBT-Bench/core/core_major_test.json"
        exit 1
    fi
    print_colored $GREEN "✅ Core beliefs dataset found"
fi

if [[ "$DATASET" == "all" || "$DATASET" == "core_fine" ]]; then
    if [[ ! -f "CBT-Bench/fine/core_fine_test.json" ]]; then
        print_colored $RED "❌ Fine-grained beliefs dataset not found: CBT-Bench/fine/core_fine_test.json"
        exit 1
    fi
    print_colored $GREEN "✅ Fine-grained beliefs dataset found"
fi

if [[ "$DATASET" == "all" || "$DATASET" == "distortions" ]]; then
    if [[ ! -f "CBT-Bench/distortions/distortions_test.json" ]]; then
        print_colored $RED "❌ Cognitive distortions dataset not found: CBT-Bench/distortions/distortions_test.json"
        exit 1
    fi
    print_colored $GREEN "✅ Cognitive distortions dataset found"
fi

# Print evaluation summary
print_colored $BLUE "📋 Evaluation Summary:"
echo "   Dataset: $DATASET"
if [[ -n "$NUM_CASES" ]]; then
    echo "   Cases per dataset: $NUM_CASES"
else
    echo "   Cases per dataset: All available"
fi

# Build command
CMD="python cbt/cbt_main.py --dataset $DATASET --mode evaluation"

if [[ -n "$NUM_CASES" ]]; then
    CMD="$CMD --num-cases $NUM_CASES"
fi

print_colored $YELLOW "🚀 Starting evaluation..."
echo "Command: $CMD"
echo ""

# Record start time
START_TIME=$(date +%s)

# Run evaluation
if $CMD; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    print_colored $GREEN "✅ Evaluation completed successfully!"
    print_colored $BLUE "⏱️  Total time: ${DURATION} seconds"
    
else
    print_colored $RED "❌ Evaluation failed!"
    exit 1
fi 