class CPUConfig:
    # Training parameters
    n_episodes = 500
    eval_interval = 50
    
    # Agent parameters
    learning_rate = 1e-3
    batch_size = 32
    memory_size = 1000
    
    # Environment parameters
    starting_stack = 200
    big_blind = 2
    
    # Device
    device = 'cpu'
    
    # Logging
    log_interval = 25
    save_results = True
    results_dir = 'results/cpu'