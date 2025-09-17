#!/bin/bash

# SUM Poker Research Framework - Virtual Environment Activation
# Quick script to activate the virtual environment

if [ ! -d "sum_poker_venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup_venv.sh first to create the environment"
    exit 1
fi

echo "🎯 Activating SUM Poker Research Framework Virtual Environment"
source sum_poker_venv/bin/activate

echo "✅ Virtual environment activated!"
echo "🐍 Python: $(which python)"
echo "📦 Pip: $(which pip)"
echo ""
echo "🚀 Ready to run experiments:"
echo "   - Validate: python run_research_pipeline.py --validate-only"
echo "   - Quick test: python run_research_pipeline.py --phases gpu_training --gpu-hands 100000"
echo "   - Full research: python run_research_pipeline.py"
echo ""
echo "💡 Use 'deactivate' to exit the virtual environment"

# Start a new shell with the virtual environment activated
exec "$SHELL"