#!/usr/bin/env python3
"""
GPU-Accelerated Strategic Uncertainty Management Experiments
Comprehensive runner for poker and Quake experiments with CUDA optimization
"""

import argparse
import os
import sys
import time
import subprocess
from datetime import datetime
from typing import Dict, List
import json

try:
    import torch
    import torch.cuda as cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. GPU acceleration disabled.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Progress bars disabled.")

def check_gpu_availability():
    """Check GPU availability and CUDA setup"""
    if not TORCH_AVAILABLE:
        return False, "PyTorch not installed"
    
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return False, "No CUDA devices found"
    
    return True, f"{gpu_count} CUDA device(s) available"

def setup_cuda_environment(device_ids: List[int] = None):
    """Setup CUDA environment variables"""
    if device_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
        print(f"CUDA devices set to: {device_ids}")
    
    # Optimize CUDA settings
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
    os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0;7.5;8.0;8.6'  # Support modern GPUs
    
    if TORCH_AVAILABLE:
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

def run_poker_experiment(args):
    """Run poker experiment with specified configuration"""
    print(f"\n🃏 Starting Poker Tournament Experiment")
    print(f"Device: {args.device}")
    print(f"Hands: {args.hands}")
    print(f"Progress: {args.progress}")
    
    cmd = [
        sys.executable, 
        'experiments/real_poker_experiments.py',
        '--device', args.device,
        '--hands', str(args.hands)
    ]
    
    if args.progress:
        cmd.append('--progress')
    
    if args.multi_gpu:
        cmd.append('--multi-gpu')
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ Poker experiment completed successfully")
            print(result.stdout)
        else:
            print("❌ Poker experiment failed")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running poker experiment: {e}")
    
    end_time = time.time()
    print(f"⏱️ Poker experiment duration: {end_time - start_time:.2f} seconds")

def run_quake_experiment(args):
    """Run Quake experiment with specified configuration"""
    print(f"\n🎮 Starting Quake Combat Experiment")
    print(f"Device: {args.device}")
    print(f"Episodes: {args.episodes}")
    print(f"Progress: {args.progress}")
    
    cmd = [
        sys.executable,
        'experiments/real_quake_experiments.py',
        '--device', args.device,
        '--episodes', str(args.episodes)
    ]
    
    if args.progress:
        cmd.append('--progress')
    
    if args.cuda_optimize:
        cmd.append('--cuda-optimize')
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ Quake experiment completed successfully")
            print(result.stdout)
        else:
            print("❌ Quake experiment failed")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running Quake experiment: {e}")
    
    end_time = time.time()
    print(f"⏱️ Quake experiment duration: {end_time - start_time:.2f} seconds")

def run_benchmark(args):
    """Run comprehensive benchmark across CPU and GPU"""
    print("\nRunning Comprehensive GPU Benchmark")
    
    benchmark_results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': get_system_info(),
        'experiments': []
    }
    
    # Test configurations
    configs = [
        {'device': 'cpu', 'hands': 100, 'episodes': 20},
    ]
    
    if args.device == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
        configs.extend([
            {'device': 'cuda', 'hands': 200, 'episodes': 50},
            {'device': 'cuda', 'hands': 500, 'episodes': 100},
        ])
    
    for config in configs:
        print(f"\nBenchmark: {config['device'].upper()} - Hands: {config['hands']}, Episodes: {config['episodes']}")
        
        # Enable progress bars for benchmarks
        args.device = config['device']
        args.hands = config['hands']
        args.progress = True  # Force progress bars for benchmarks
        
        start_time = time.time()
        run_poker_experiment(args)
        poker_time = time.time() - start_time
        
        # Quake benchmark
        args.episodes = config['episodes']
        start_time = time.time()
        run_quake_experiment(args)
        quake_time = time.time() - start_time
        
        benchmark_results['experiments'].append({
            'config': config,
            'poker_time': poker_time,
            'quake_time': quake_time,
            'total_time': poker_time + quake_time
        })
    
    # Save benchmark results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_file = f"results/gpu_benchmark_{timestamp}.json"
    os.makedirs('results', exist_ok=True)
    
    with open(benchmark_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"\n📈 Benchmark results saved to: {benchmark_file}")
    print_benchmark_summary(benchmark_results)

def get_system_info():
    """Get system information for benchmarking"""
    info = {
        'python_version': sys.version,
        'platform': sys.platform
    }
    
    if TORCH_AVAILABLE:
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    
    return info

def print_benchmark_summary(results):
    """Print benchmark summary"""
    print("\nBENCHMARK SUMMARY")
    print("=" * 50)
    
    for exp in results['experiments']:
        config = exp['config']
        print(f"\n{config['device'].upper()} Configuration:")
        print(f"  Poker ({config['hands']} hands): {exp['poker_time']:.2f}s")
        print(f"  Quake ({config['episodes']} episodes): {exp['quake_time']:.2f}s")
        print(f"  Total: {exp['total_time']:.2f}s")
    
    # Calculate speedup if both CPU and GPU results exist
    cpu_results = [exp for exp in results['experiments'] if exp['config']['device'] == 'cpu']
    gpu_results = [exp for exp in results['experiments'] if exp['config']['device'] == 'cuda']
    
    if cpu_results and gpu_results:
        cpu_time = cpu_results[0]['total_time']
        gpu_time = gpu_results[0]['total_time']
        speedup = cpu_time / gpu_time
        print(f"\nGPU Speedup: {speedup:.2f}x faster than CPU")

def setup_git_repository():
    """Setup git repository with proper configuration"""
    print("\n📦 Setting up Git Repository")
    
    commands = [
        ['git', 'init'],
        ['git', 'add', '.'],
        ['git', 'commit', '-m', 'Initial commit: Strategic Uncertainty Management with GPU acceleration'],
        ['git', 'branch', '-M', 'main']
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode == 0:
                print(f"✅ {' '.join(cmd)}")
            else:
                print(f"❌ {' '.join(cmd)}: {result.stderr}")
        except Exception as e:
            print(f"❌ Error running {' '.join(cmd)}: {e}")
    
    print("\n📝 Git repository initialized. To push to remote:")
    print("git remote add origin <your-repo-url>")
    print("git push -u origin main")

def main():
    parser = argparse.ArgumentParser(
        description='GPU-Accelerated Strategic Uncertainty Management Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run poker on GPU with progress
  python run_gpu_experiments.py --experiment poker --device cuda --hands 500 --progress
  
  # Run Quake with CUDA optimization
  python run_gpu_experiments.py --experiment quake --device cuda --episodes 100 --cuda-optimize
  
  # Run comprehensive benchmark
  python run_gpu_experiments.py --benchmark --device cuda
  
  # Setup git repository
  python run_gpu_experiments.py --setup-git
        """
    )
    
    parser.add_argument('--experiment', choices=['poker', 'quake', 'both'], default='both',
                       help='Which experiment to run')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda',
                       help='Device to use for computation')
    parser.add_argument('--hands', type=int, default=200,
                       help='Number of poker hands to play')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of Quake episodes to run')
    parser.add_argument('--progress', action='store_true',
                       help='Show progress bars')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use multiple GPUs for poker')
    parser.add_argument('--cuda-optimize', action='store_true',
                       help='Enable CUDA optimizations for Quake')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=None,
                       help='Specific GPU IDs to use')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run comprehensive benchmark')
    parser.add_argument('--setup-git', action='store_true',
                       help='Setup git repository')
    
    args = parser.parse_args()
    
    print("GPU-Accelerated Strategic Uncertainty Management")
    print("=" * 55)
    
    # Setup git if requested
    if args.setup_git:
        setup_git_repository()
        return
    
    # Check GPU availability
    if args.device == 'cuda':
        gpu_available, gpu_info = check_gpu_availability()
        if gpu_available:
            print(f"GPU Status: {gpu_info}")
            setup_cuda_environment(args.gpu_ids)
        else:
            print(f"GPU Status: {gpu_info}")
            print("Falling back to CPU mode")
            args.device = 'cpu'
    
    # Run benchmark if requested
    if args.benchmark:
        run_benchmark(args)
        return
    
    # Run experiments
    start_time = time.time()
    
    if args.experiment in ['poker', 'both']:
        run_poker_experiment(args)
    
    if args.experiment in ['quake', 'both']:
        run_quake_experiment(args)
    
    total_time = time.time() - start_time
    print(f"\n🎉 All experiments completed in {total_time:.2f} seconds")
    
    # Show results directory
    if os.path.exists('results'):
        print(f"\n📁 Results saved in: {os.path.abspath('results')}")
        result_files = [f for f in os.listdir('results') if f.endswith(('.json', '.png', '.pdf'))]
        if result_files:
            print("Latest results:")
            for f in sorted(result_files)[-5:]:  # Show last 5 files
                print(f"  - {f}")

if __name__ == '__main__':
    main()