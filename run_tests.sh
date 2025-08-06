#!/bin/bash

# Complete test suite for Dense-Vector DSPy Agent

echo "================================================================"
echo "Dense-Vector DSPy Agent - Complete Test Suite"
echo "================================================================"

# Activate virtual environment
source .venv/bin/activate

# Stage 1: Basic Components
echo -e "\n[Stage 1/6] Basic Component Testing..."
python test_basic.py
if [ $? -ne 0 ]; then
    echo "❌ Stage 1 failed"
    exit 1
fi

# Stage 2: Integration
echo -e "\n[Stage 2/6] Integration Testing..."
python test_integration.py
if [ $? -ne 0 ]; then
    echo "❌ Stage 2 failed"
    exit 1
fi

# Stage 3: Small Training
echo -e "\n[Stage 3/6] Small Training Test..."
python test_small_train.py
if [ $? -ne 0 ]; then
    echo "❌ Stage 3 failed"
    exit 1
fi

# Stage 4: Groq API
echo -e "\n[Stage 4/6] Groq API Testing..."
python test_groq.py
if [ $? -ne 0 ]; then
    echo "❌ Stage 4 failed"
    exit 1
fi

echo -e "\n================================================================"
echo "✅ All basic tests passed!"
echo "================================================================"

# Instructions for full-scale testing
echo -e "\nNext steps for full testing:"
echo ""
echo "Stage 5 - Medium Scale Test (30-45 min):"
echo "  python train.py --train-size 50 --dev-size 100 --epochs 2"
echo "  python eval.py --eval-size 100 --save-results medium_results.json"
echo ""
echo "Stage 6 - Full Scale Test (2-3 hours):"
echo "  python train.py --train-size 200 --dev-size 500 --epochs 3"
echo "  python eval.py --eval-size 500 --save-results final_results.json"
echo ""
echo "Interactive testing:"
echo "  python run.py                    # Dense mode"
echo "  python run.py --mode prompt      # Baseline mode"
echo "  python run.py --use-groq         # With Groq decoding"
echo ""