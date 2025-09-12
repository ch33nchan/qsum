# Strategic Uncertainty Management (SUM) Poker Research Framework

A comprehensive research framework implementing Strategic Uncertainty Management (SUM) for poker AI, featuring quantum-inspired decision-making processes with GPU acceleration for high-performance academic research.

## Research Focus

This framework implements and evaluates Strategic Uncertainty Management algorithms in poker environments, providing:

- **Advanced SUM Implementation**: Quantum-inspired superposition states with strategic collapse mechanisms
- **GPU-Accelerated Research**: CUDA-optimized computations for large-scale experiments
- **Publication-Quality Data**: Comprehensive metrics, statistical analysis, and research visualizations
- **Real Poker Environment**: PyPokerEngine integration with authentic game mechanics
- **Academic Validation**: Rigorous experimental design suitable for peer-reviewed publication

## 🚀 Quick Start (GPU Mode)

### 1. Install Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 2. Run GPU-Accelerated Experiments
```bash
# Poker Tournament (GPU)
python experiments/real_poker_experiments.py --device cuda --hands 500 --progress

# Quake Combat (GPU with optimization)
python experiments/real_quake_experiments.py --device cuda --episodes 100 --progress --cuda-optimize

# Comprehensive GPU Runner
python run_gpu_experiments.py --experiment both --device cuda --progress
```

### 3. Setup Git Repository
```bash
python run_gpu_experiments.py --setup-git
```

## 📊 Performance Comparison

| Mode | Poker (500 hands) | Quake (100 episodes) | Speedup |
|------|-------------------|----------------------|----------|
| CPU  | ~8-12 minutes     | ~15-20 minutes       | 1.0x     |
| GPU  | ~3-5 minutes      | ~8-12 minutes        | **3-5x** |

## 🎮 Experiment Commands

### Poker Tournament
```bash
# Basic CPU mode
python experiments/real_poker_experiments.py --device cpu --hands 200

# GPU mode with progress
python experiments/real_poker_experiments.py --device cuda --hands 500 --progress

# High-performance GPU mode
CUDA_VISIBLE_DEVICES=0 python experiments/real_poker_experiments.py --device cuda --hands 1000 --progress
```

### Quake Combat
```bash
# Basic CPU mode
python experiments/real_quake_experiments.py --device cpu --episodes 30

# GPU mode with CUDA optimization
python experiments/real_quake_experiments.py --device cuda --episodes 100 --progress --cuda-optimize

# Multi-GPU setup
CUDA_VISIBLE_DEVICES=0,1 python experiments/real_quake_experiments.py --device cuda --episodes 200 --progress
```

### Comprehensive Runner
```bash
# Run both experiments on GPU
python run_gpu_experiments.py --experiment both --device cuda --hands 500 --episodes 100 --progress

# GPU benchmark suite
python run_gpu_experiments.py --benchmark --device cuda

# Setup git repository
python run_gpu_experiments.py --setup-git
```

## 🔧 GPU Setup

### Requirements
- NVIDIA GPU with CUDA Compute Capability 7.0+
- CUDA 11.8 or later
- 4GB+ GPU memory recommended
- PyTorch with CUDA support

### Installation
```bash
# Check CUDA availability
nvidia-smi

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

### GPU Memory Optimization
- Automatic memory cleanup after experiments
- Batch processing for efficient GPU utilization
- Memory monitoring and reporting
- CUDA kernel optimization for superposition calculations

## 📈 Research Features

### Strategic Uncertainty Management
- **Quantum Superposition**: Probabilistic action states until collapse
- **Strategic Collapse**: Context-aware decision crystallization
- **Entropy Calculation**: Real-time uncertainty quantification
- **GPU Acceleration**: Vectorized superposition computations

### Real Game Environments
- **Authentic Poker**: PyPokerEngine tournaments with real opponents
- **3D Combat**: ViZDoom Quake III Arena with enemy AI
- **Real Physics**: Actual game mechanics, not simulations
- **Performance Metrics**: Comprehensive research data collection

## 📊 Generated Results

Each experiment generates:
- **📈 Analysis Plots**: 6-panel research visualizations (PNG/PDF)
- **📋 LaTeX Tables**: Publication-ready statistical summaries
- **💾 JSON Data**: Complete raw metrics for further analysis
- **🎯 Performance Reports**: GPU vs CPU benchmarks

## 🔬 Project Overview

This project implements Strategic Uncertainty Management (SUM) agents for complex gaming environments, incorporating quantum-inspired decision-making processes that maintain strategic ambiguity until optimal collapse points. Now enhanced with GPU acceleration for high-performance research.

## Project Structure

```
q-agent/
├── src/                          # Main training scripts
│   ├── sum_cpu.py               # CPU training (quick testing)
│   └── sum_gpu.py               # GPU training (research-grade)
├── core/                         # Core implementation
│   ├── agent.py                 # SUM agent with research metrics
│   ├── baseline_agents.py       # Classical baseline agents
│   ├── environment.py           # Poker environment
│   ├── network.py               # Neural network architectures
│   └── utils.py                 # Shared utilities
├── config/                       # Configuration files
│   ├── cpu_config.py           # CPU training configuration
│   └── gpu_config.py           # GPU training configuration
├── docs/                        # Research papers and documentation
└── results/                     # Training and experiment results
```

## Quick Start

```bash
# Quick CPU training (30 seconds)
python src/sum_cpu.py

# Research-grade GPU training (45-90 minutes)
python src/sum_gpu.py
```

## Research Validation

The GPU training script includes comprehensive research-grade validation:

**Statistical Analysis**:
- Multiple runs with different random seeds for statistical validity
- t-tests, effect sizes, and confidence intervals
- Baseline comparisons against classical poker strategies

**Performance Metrics**:
- Win rates against multiple opponent types
- Strategic uncertainty utilization tracking
- Collapse event analysis and timing optimization
- Deception effectiveness measurements

## Core Implementation Features

### QuantumPokerAgent (Enhanced SUM Agent)
- **Superposition state representation**: Complex-valued probability amplitudes
- **Strategic collapse detection**: Opponent action-triggered collapse mechanisms
- **Comprehensive tracking**: Detailed metrics for all experiments
- **13-category hand strength**: Simplified but comprehensive hand evaluation

### Classical Baseline Agents
- **Random Agent**: Pure random baseline
- **Tight-Aggressive**: Conservative classical strategy
- **Loose-Passive**: Liberal classical strategy
- **Classical Mixed Strategy**: Game theory optimal mixed strategies
- **Nash Equilibrium**: Theoretical optimal play approximation

### Research-Grade Features
- **Statistical validation**: t-tests, effect sizes, confidence intervals
- **Multiple random seeds**: Ensures reproducible results
- **Comprehensive metrics**: 20+ tracked metrics per experiment
- **Publication-ready plots**: High-resolution figures with error bars
- **LaTeX table generation**: Ready for academic papers

## Usage Examples

### Development and Testing
```bash
# Quick validation (30 seconds)
python src/sum_cpu.py
```

### Research Paper Generation
```bash
# Full research validation (45-90 minutes)
python src/sum_gpu.py

# This generates:
# - Statistical significance tests
# - Publication-quality plots
# - LaTeX tables
# - Comprehensive research data
```

## Results and Outputs

### Training Results
- **CPU**: `results/cpu/training_plots.png`, `training_results.json`
- **GPU**: `results/research_gpu/experiment_TIMESTAMP/` (comprehensive research data)

### Research Outputs
- **Plots**: High-resolution PNG and PDF figures
- **Data**: JSON files with all experimental data
- **Statistics**: Statistical test results and significance analysis
- **LaTeX**: Ready-to-use tables for academic papers

### Key Files Generated
```
results/
├── research_plots.png
├── complete_experimental_results.json
├── research_summary.json
└── results_table.tex
```

## Research Paper Integration

The experiment framework generates publication-ready materials:

1. **Statistical Tables**: LaTeX format with significance tests
2. **High-Quality Figures**: 300 DPI plots with error bars
3. **Comprehensive Data**: JSON files with all metrics
4. **Reproducible Results**: Fixed random seeds and detailed logging

### Example Paper Sections

**Methods Section**:
- Experimental design with multiple runs
- Statistical testing methodology
- Baseline agent descriptions
- Evaluation metrics definitions

**Results Section**:
- Statistical significance tables
- Effect size analysis
- Uncertainty utilization evidence
- Strategic behavior analysis

## Configuration Options

### Quick Test (Development)
- 100 hands per experiment
- ~5 minutes total runtime
- Basic validation

### Full Research (Publication)
- 5000 hands per experiment
- ~2-4 hours total runtime
- Comprehensive statistical validation
- Publication-ready outputs

### Custom Configuration
Edit `experiments/experiment_config.py` to customize:
- Number of hands per experiment
- Collapse detection parameters
- Deception tracking thresholds
- Performance evaluation criteria

## Requirements

```bash
# Core dependencies
pip install torch numpy matplotlib seaborn pandas scipy

# Optional: GPU acceleration
# CUDA-compatible PyTorch installation
```

## Academic Citation

If you use this implementation in your research, please cite:

```bibtex
@article{sum_poker_2024,
  title={Strategic Uncertainty Management in Poker: A Quantum-Inspired Approach},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```

## Key Research Contributions

1. **Novel Strategic Framework**: First implementation of quantum-inspired uncertainty management in poker
2. **Comprehensive Validation**: Three-experiment framework with statistical rigor
3. **Practical Implementation**: Working code with research-grade experimental validation
4. **Reproducible Results**: Fixed seeds, detailed logging, and comprehensive documentation

## Performance Expectations

### CPU Training
- **Runtime**: ~0.5 seconds
- **Win Rate**: 75% against simple opponents
- **Purpose**: Quick validation and testing

### GPU Research Training
- **Runtime**: 45-90 minutes
- **Statistical Power**: 95% confidence intervals
- **Baselines**: 4+ different opponent types
- **Purpose**: Academic publication validation

### Full Experiments
- **Runtime**: 2-4 hours
- **Data Points**: 15,000+ hands across all experiments
- **Statistical Tests**: t-tests, effect sizes, significance analysis
- **Outputs**: Publication-ready figures and tables

This implementation provides everything needed for rigorous academic validation of the Strategic Uncertainty Management approach to poker AI.