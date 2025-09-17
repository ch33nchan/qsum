#!/bin/bash

# SUM Poker Research Framework - Virtual Environment Setup Script
# This script creates a clean virtual environment and installs all dependencies

set -e  # Exit on any error

echo "🎯 Setting up SUM Poker Research Framework Virtual Environment"
echo "================================================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "🐍 Found Python $PYTHON_VERSION"

# Check if virtual environment already exists
if [ -d "sum_poker_venv" ]; then
    echo "📁 Virtual environment already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf sum_poker_venv
    else
        echo "✅ Using existing virtual environment"
        source sum_poker_venv/bin/activate
        echo "🎯 Virtual environment activated: $(which python)"
        exit 0
    fi
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv sum_poker_venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source sum_poker_venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install torch torchvision numpy scipy scikit-learn matplotlib seaborn pandas plotly tqdm

# Install poker engine
echo "🎮 Installing poker engine..."
pip install PyPokerEngine==1.0.1

# Install development tools
echo "🛠️  Installing development tools..."
pip install jupyter ipython pytest rich psutil

# Install additional dependencies
echo "📊 Installing additional analysis tools..."
pip install numba statsmodels opencv-python h5py openpyxl xlsxwriter pyarrow memory-profiler

# Install code quality tools
echo "✨ Installing code quality tools..."
pip install black flake8 mypy

# Install configuration and async tools
echo "⚙️  Installing configuration tools..."
pip install pyyaml click aiohttp uvloop networkx bokeh

# Install experiment tracking (optional)
echo "📈 Installing experiment tracking tools..."
pip install wandb tensorboard

echo ""
echo "✅ Virtual environment setup complete!"
echo ""
echo "🚀 To get started:"
echo "   1. Activate the environment: source sum_poker_venv/bin/activate"
echo "   2. Validate the setup: python run_research_pipeline.py --validate-only"
echo "   3. Run a quick test: python run_research_pipeline.py --phases gpu_training --gpu-hands 100000"
echo "   4. Run full research: python run_research_pipeline.py"
echo ""
echo "📁 Virtual environment location: $(pwd)/sum_poker_venv"
echo "🐍 Python executable: $(pwd)/sum_poker_venv/bin/python"
echo ""
echo "💡 Tips:"
echo "   - Always activate the environment before running experiments"
echo "   - Use 'deactivate' to exit the virtual environment"
echo "   - The environment is self-contained and portable"
echo ""
echo "🎯 Ready to run SOTA poker AI research!"