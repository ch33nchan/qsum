#!/usr/bin/env python3
"""
Fast GPU Experiments - Lightweight version for quick testing
Skips ViZDoom and uses minimal configurations for speed
"""

import argparse
import os
import sys
import time
import subprocess
from datetime import datetime

def run_fast_poker_only(device='cuda', hands=50):
    """Run only poker experiment with minimal hands"""
    print(f"\nFast Poker Test - {device.upper()} mode")
    print(f"Running {hands} hands for quick validation")
    
    cmd = [
        sys.executable, 
        'experiments/real_poker_experiments.py',
        '--device', device,
        '--hands', str(hands),
        '--progress'
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    duration = time.time() - start_time
    
    if result.returncode == 0:
        print(f"Poker test completed in {duration:.2f} seconds")
        print("Last few lines of output:")
        print('\n'.join(result.stdout.split('\n')[-10:]))
    else:
        print(f"Poker test failed: {result.stderr}")
    
    return result.returncode == 0

def run_cpu_vs_gpu_comparison():
    """Quick CPU vs GPU comparison with minimal data"""
    print("\n=== FAST CPU vs GPU COMPARISON ===")
    
    results = {}
    
    # CPU test
    print("\nTesting CPU performance...")
    start_time = time.time()
    cpu_success = run_fast_poker_only('cpu', 25)
    results['cpu_time'] = time.time() - start_time
    results['cpu_success'] = cpu_success
    
    # GPU test
    print("\nTesting GPU performance...")
    start_time = time.time()
    gpu_success = run_fast_poker_only('cuda', 25)
    results['gpu_time'] = time.time() - start_time
    results['gpu_success'] = gpu_success
    
    # Summary
    print("\n=== FAST COMPARISON RESULTS ===")
    if results['cpu_success']:
        print(f"CPU (25 hands): {results['cpu_time']:.2f} seconds")
    else:
        print("CPU test failed")
        
    if results['gpu_success']:
        print(f"GPU (25 hands): {results['gpu_time']:.2f} seconds")
        if results['cpu_success']:
            speedup = results['cpu_time'] / results['gpu_time']
            print(f"GPU Speedup: {speedup:.2f}x faster")
    else:
        print("GPU test failed")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Fast GPU Experiments - Quick Testing')
    parser.add_argument('--mode', choices=['poker-only', 'comparison', 'gpu-test'], 
                       default='comparison', help='Test mode to run')
    parser.add_argument('--hands', type=int, default=50, 
                       help='Number of poker hands (default: 50 for speed)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda',
                       help='Device for single tests')
    
    args = parser.parse_args()
    
    print("Fast GPU Experiments - Quick Testing")
    print("=" * 40)
    print(f"Mode: {args.mode}")
    
    if args.mode == 'poker-only':
        run_fast_poker_only(args.device, args.hands)
    elif args.mode == 'comparison':
        run_cpu_vs_gpu_comparison()
    elif args.mode == 'gpu-test':
        print(f"\nQuick GPU validation with {args.hands} hands")
        success = run_fast_poker_only('cuda', args.hands)
        if success:
            print("\nGPU test PASSED - GPU acceleration working")
        else:
            print("\nGPU test FAILED - Check CUDA setup")
    
    print("\nFast experiments completed!")
    print("For full benchmarks, use: python run_gpu_experiments.py --benchmark --device cuda")

if __name__ == '__main__':
    main()