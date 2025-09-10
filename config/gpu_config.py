import torch

class GPUConfig:
    # Research-grade training parameters
    n_episodes = 10000  # Increased for statistical significance
    eval_interval = 500
    n_evaluation_runs = 5  # Multiple runs for statistical validity
    
    # Agent parameters
    learning_rate = 1e-4
    batch_size = 256
    memory_size = 50000
    
    # Environment parameters
    starting_stack = 200
    big_blind = 2
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Research metrics and validation
    statistical_significance_tests = True
    confidence_level = 0.95
    min_sample_size = 1000  # Minimum games for statistical validity
    
    # Baseline comparisons for research validation
    baseline_agents = {
        'random': True,
        'tight_aggressive': True,
        'loose_passive': True,
        'classical_mixed_strategy': True
    }
    
    # Advanced metrics for research paper
    track_uncertainty_metrics = True
    track_strategic_diversity = True
    track_collapse_patterns = True
    track_deception_effectiveness = True
    
    # Logging
    log_interval = 250
    save_results = True
    results_dir = 'results/research_gpu'
    detailed_logging = True
    
    # GPU specific
    use_multi_gpu = True
    mixed_precision = True
    
    # Research reproducibility
    random_seeds = [42, 123, 456, 789, 999]  # Multiple seeds for robustness
    save_model_checkpoints = True
    checkpoint_interval = 2000