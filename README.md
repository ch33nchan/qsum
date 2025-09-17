# SUM Poker Agent: Strategic Uncertainty Management Research Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art research framework implementing Strategic Uncertainty Management (SUM) for poker agents, featuring novel neural architectures, multi-objective loss functions, and comprehensive experimental protocols.

## 🎯 Overview

This repository implements the SUM (Strategic Uncertainty Management) poker agent, a novel approach to poker AI that maintains multiple strategic options in superposition until committing to specific actions based on information-theoretic principles. The framework includes:

- **Novel Neural Architecture**: GameEncoder, HistoryEncoder, StrategyHead, WeightHead, and CommitmentHead
- **Multi-Objective Loss Function**: Strategy optimization, commitment timing, and deception rewards
- **Comprehensive Benchmarking**: Against CFR, Deep CFR, NFSP, Pluribus-style, and commercial bots
- **GPU-Accelerated Training**: Parallel self-play with experience replay
- **Rigorous Evaluation**: Statistical analysis, ablation studies, and robustness testing

## 📁 Project Structure

```
q-agent/
├── agents/                     # Agent implementations
│   ├── sum_agent.py           # Main SUM agent
│   └── baseline_agents.py     # Benchmark agents (CFR, Deep CFR, etc.)
├── analysis/                   # Research analysis tools
│   └── research_analyzer.py   # Data analysis and visualization
├── docs/                       # Research documentation
│   ├── lossfunk_1.pdf        # Research paper
│   ├── lossfunk_1 (1).pdf    # Additional documentation
│   └── plan.txt              # Research plan
├── environments/               # Poker environments
│   └── poker_environment.py   # PyPokerEngine integration
├── evaluation/                 # Evaluation systems
│   └── benchmark_system.py    # Comprehensive benchmarking
├── experiments/                # Experimental frameworks
│   ├── cpu_experiments.py     # CPU-based ablation studies
│   └── real_poker_experiments.py # Legacy experiments
├── sum/                        # Core SUM algorithm
│   ├── neural_architecture.py # Neural network components
│   └── loss_functions.py      # Multi-objective loss functions
├── training/                   # Training infrastructure
│   └── self_play_trainer.py   # GPU-accelerated self-play
├── run_research_pipeline.py   # Main research execution script
└── requirements.txt           # Dependencies
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/q-agent.git
   cd q-agent
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Complete Research Pipeline

**Full Research Study (3-4 hours):**
```bash
python run_research_pipeline.py
```

**GPU Training Only:**
```bash
python run_research_pipeline.py --phases gpu_training
```

**CPU Experiments Only:**
```bash
python run_research_pipeline.py --phases cpu_experiments
```

**Custom Configuration:**
```bash
python run_research_pipeline.py --config research_config.json --gpu-hands 5000000
```

### Environment Validation

```bash
python run_research_pipeline.py --validate-only
```

## 🧠 SUM Algorithm Architecture

### Core Components

1. **GameEncoder**: Processes current game state (cards, positions, stacks, pot)
2. **HistoryEncoder**: LSTM-based action sequence encoding
3. **StrategyHead**: Generates multiple strategic options in parallel
4. **WeightHead**: Computes strategy mixing weights
5. **CommitmentHead**: Determines when to collapse superposition

### Multi-Objective Loss Function

```python
L_total = L_strategy + λ_commitment * L_commitment + λ_deception * L_deception
```

- **Strategy Loss**: Cross-entropy for action prediction
- **Commitment Loss**: Information-theoretic timing optimization
- **Deception Loss**: Opponent confusion and bluff success rewards

## 🔬 Experimental Framework

### Phase 1: GPU Training (Self-Play)
- **Duration**: 2-3 hours
- **Hands**: 10M+ poker hands
- **Parallel Games**: 64 simultaneous environments
- **Experience Replay**: Prioritized sampling
- **Target Networks**: Periodic updates for stability

### Phase 2: CPU Experiments (Ablation Studies)
- **Ablation Studies**: Commitment vs. Deception mechanisms
- **Hyperparameter Sweeps**: Strategy count, loss weights, learning rates
- **Robustness Testing**: Different stack sizes and game conditions

### Phase 3: Benchmark Gauntlet
- **Opponents**: CFR, Deep CFR, NFSP, Pluribus-style, Commercial bots
- **Statistical Analysis**: Confidence intervals, significance testing
- **Performance Metrics**: mBB/100, win rates, effect sizes

### Phase 4: Analysis and Visualization
- **Publication-Quality Plots**: Performance charts, sensitivity analysis
- **LaTeX Tables**: Statistical results for academic papers
- **Comprehensive Reports**: Markdown summaries with insights

## 📊 Results and Analysis

After running experiments, results are automatically organized:

```
research_results/session_YYYYMMDD_HHMMSS/
├── checkpoints/               # Trained model checkpoints
├── training_logs/            # Training progress logs
├── cpu_experiments/          # Ablation study results
├── benchmark_results/        # Gauntlet performance data
├── analysis/                 # Generated plots and tables
│   ├── plots/               # PNG visualizations
│   ├── latex_tables/        # Academic table formatting
│   └── research_analysis_report.md
└── RESEARCH_REPORT.md        # Executive summary
```

### Key Performance Metrics

- **mBB/100**: Milli-big blinds per 100 hands (profit measure)
- **Win Rate**: Percentage of profitable sessions
- **Statistical Significance**: p-values and effect sizes
- **Commitment Rate**: Frequency of strategy collapse
- **Deception Success**: Bluff and value bet effectiveness

## 🛠 Advanced Usage

### Custom Training Configuration

Create `research_config.json`:
```json
{
  "device": "cuda",
  "gpu_training": {
    "total_hands": 20000000,
    "parallel_games": 128,
    "batch_size": 512,
    "num_strategies": 12,
    "lambda_commitment": 0.4,
    "lambda_deception": 0.15
  },
  "phases_to_run": ["gpu_training", "benchmark_gauntlet", "analysis"]
}
```

### Individual Component Testing

**Train SUM Agent:**
```python
from agents.sum_agent import SUMAgent
from training.self_play_trainer import SelfPlayTrainer, TrainingConfig

agent = SUMAgent(name="Test_SUM", device="cuda")
config = TrainingConfig(total_hands=1_000_000)
trainer = SelfPlayTrainer(config)
results = trainer.train()
```

**Run Benchmark:**
```python
from evaluation.benchmark_system import BenchmarkGauntlet, BenchmarkConfig
from agents.baseline_agents import BaselineAgentFactory

config = BenchmarkConfig(hands_per_match=100_000)
gauntlet = BenchmarkGauntlet(config)
results = gauntlet.run_gauntlet(trained_agent)
```

**Analyze Results:**
```python
from analysis.research_analyzer import ResearchDataAnalyzer

analyzer = ResearchDataAnalyzer("research_results/session_20240115_143022")
report = analyzer.generate_comprehensive_report()
```

## 🔧 Development

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .

# Run tests
pytest tests/ -v --cov=.
```

### Adding New Baseline Agents

1. Inherit from `BasePokerPlayer`
2. Implement required methods
3. Add to `BaselineAgentFactory`
4. Update benchmark configuration

### Extending SUM Architecture

1. Modify `SUMNeuralArchitecture` in `sum/neural_architecture.py`
2. Update loss functions in `sum/loss_functions.py`
3. Adjust training configuration
4. Run ablation studies to validate improvements

## 📈 Performance Expectations

### Hardware Requirements

**Minimum (CPU-only):**
- 8GB RAM
- 4 CPU cores
- 10GB disk space

**Recommended (GPU):**
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- 50GB disk space
- CUDA 12.0+

### Timing Estimates

| Phase | CPU (8 cores) | GPU (RTX 4090) |
|-------|---------------|----------------|
| Training (10M hands) | 8-12 hours | 2-3 hours |
| CPU Experiments | 4-6 hours | N/A |
| Benchmark Gauntlet | 2-3 hours | 30-45 minutes |
| Analysis | 10-15 minutes | 5-10 minutes |
| **Total** | **14-21 hours** | **3-4 hours** |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Research Contributions

- Novel neural architectures for poker AI
- Improved loss function formulations
- Additional baseline agent implementations
- Enhanced evaluation metrics
- Optimization improvements

## 📚 Citation

If you use this research framework in your work, please cite:

```bibtex
@article{sum_poker_2024,
  title={Strategic Uncertainty Management in Poker: A Novel Neural Architecture for Multi-Agent Game Theory},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024},
  note={Available at: https://github.com/your-username/q-agent}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyPokerEngine for poker environment simulation
- PyTorch team for deep learning framework
- Poker AI research community for foundational work
- Contributors and beta testers

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/q-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/q-agent/discussions)
- **Email**: your.email@domain.com

---

**Built with ❤️ for advancing poker AI research**