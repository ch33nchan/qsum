#!/usr/bin/env python3
"""
Strategic Uncertainty Management (SUM) Poker Research Framework
Comprehensive poker experiments for academic publication
"""

import argparse
import os
import sys
import time
import subprocess
import json
from datetime import datetime
from typing import Dict, List

def run_poker_experiment(config: Dict) -> Dict:
    """Run a single poker experiment configuration"""
    print(f"\nRunning: {config['name']}")
    print(f"Device: {config['device']}, Hands: {config['hands']}, Opponents: {config['opponents']}")
    
    cmd = [
        sys.executable, 
        'experiments/real_poker_experiments.py',
        '--device', config['device'],
        '--hands', str(config['hands']),
        '--progress'
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    duration = time.time() - start_time
    
    success = result.returncode == 0
    
    # Extract results from output
    output_lines = result.stdout.split('\n')
    win_rate = 0.0
    total_winnings = 0
    hands_per_second = 0.0
    
    for line in output_lines:
        if 'Win rate:' in line:
            win_rate = float(line.split(':')[1].strip().replace('%', ''))
        elif 'Total winnings:' in line:
            total_winnings = int(line.split(':')[1].strip())
        elif 'Performance:' in line and 'hands/second' in line:
            hands_per_second = float(line.split(':')[1].strip().split()[0])
    
    return {
        'config': config,
        'success': success,
        'duration': duration,
        'win_rate': win_rate,
        'total_winnings': total_winnings,
        'hands_per_second': hands_per_second,
        'output': result.stdout if success else result.stderr
    }

def run_comprehensive_study():
    """Run comprehensive poker study for publication"""
    print("\n=== STRATEGIC UNCERTAINTY MANAGEMENT POKER RESEARCH ===")
    print("Comprehensive study for academic publication")
    
    # Research configurations
    configs = [
        # Baseline studies
        {'name': 'CPU_Baseline_Small', 'device': 'cpu', 'hands': 100, 'opponents': 'standard'},
        {'name': 'CPU_Baseline_Medium', 'device': 'cpu', 'hands': 500, 'opponents': 'standard'},
        {'name': 'CPU_Baseline_Large', 'device': 'cpu', 'hands': 1000, 'opponents': 'standard'},
        
        # GPU acceleration studies
        {'name': 'GPU_Small_Scale', 'device': 'cuda', 'hands': 100, 'opponents': 'standard'},
        {'name': 'GPU_Medium_Scale', 'device': 'cuda', 'hands': 500, 'opponents': 'standard'},
        {'name': 'GPU_Large_Scale', 'device': 'cuda', 'hands': 1000, 'opponents': 'standard'},
        {'name': 'GPU_Publication_Scale', 'device': 'cuda', 'hands': 2000, 'opponents': 'standard'},
        
        # Performance scaling studies
        {'name': 'GPU_Ultra_Scale', 'device': 'cuda', 'hands': 5000, 'opponents': 'standard'},
        {'name': 'GPU_Research_Scale', 'device': 'cuda', 'hands': 10000, 'opponents': 'standard'},
    ]
    
    results = []
    total_configs = len(configs)
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{total_configs}] {config['name']}")
        print("-" * 50)
        
        result = run_poker_experiment(config)
        results.append(result)
        
        if result['success']:
            print(f"SUCCESS: Win rate {result['win_rate']:.1f}%, "
                  f"Winnings: {result['total_winnings']}, "
                  f"Speed: {result['hands_per_second']:.1f} hands/s")
        else:
            print(f"FAILED: {result['output'][:200]}...")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/poker_research_study_{timestamp}.json"
    
    os.makedirs('results', exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'study_type': 'comprehensive_poker_research',
            'total_experiments': len(results),
            'successful_experiments': sum(1 for r in results if r['success']),
            'results': results
        }, f, indent=2)
    
    print(f"\n=== RESEARCH STUDY COMPLETED ===")
    print(f"Results saved to: {results_file}")
    
    # Generate summary
    generate_research_summary(results, timestamp)
    
    return results

def generate_research_summary(results: List[Dict], timestamp: str):
    """Generate publication-ready research summary"""
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful experiments to summarize")
        return
    
    # Performance analysis
    cpu_results = [r for r in successful_results if r['config']['device'] == 'cpu']
    gpu_results = [r for r in successful_results if r['config']['device'] == 'cuda']
    
    summary = {
        'study_overview': {
            'total_experiments': len(results),
            'successful_experiments': len(successful_results),
            'cpu_experiments': len(cpu_results),
            'gpu_experiments': len(gpu_results)
        },
        'performance_metrics': {},
        'scalability_analysis': {},
        'sum_effectiveness': {}
    }
    
    # CPU vs GPU comparison
    if cpu_results and gpu_results:
        avg_cpu_speed = sum(r['hands_per_second'] for r in cpu_results) / len(cpu_results)
        avg_gpu_speed = sum(r['hands_per_second'] for r in gpu_results) / len(gpu_results)
        speedup = avg_gpu_speed / avg_cpu_speed if avg_cpu_speed > 0 else 0
        
        summary['performance_metrics'] = {
            'avg_cpu_speed': avg_cpu_speed,
            'avg_gpu_speed': avg_gpu_speed,
            'gpu_speedup': speedup,
            'cpu_win_rates': [r['win_rate'] for r in cpu_results],
            'gpu_win_rates': [r['win_rate'] for r in gpu_results]
        }
    
    # Scalability analysis
    gpu_by_scale = {}
    for result in gpu_results:
        hands = result['config']['hands']
        if hands not in gpu_by_scale:
            gpu_by_scale[hands] = []
        gpu_by_scale[hands].append(result)
    
    scalability = {}
    for hands, results_list in gpu_by_scale.items():
        avg_speed = sum(r['hands_per_second'] for r in results_list) / len(results_list)
        avg_win_rate = sum(r['win_rate'] for r in results_list) / len(results_list)
        scalability[hands] = {
            'avg_speed': avg_speed,
            'avg_win_rate': avg_win_rate,
            'experiments': len(results_list)
        }
    
    summary['scalability_analysis'] = scalability
    
    # SUM effectiveness
    all_win_rates = [r['win_rate'] for r in successful_results]
    summary['sum_effectiveness'] = {
        'overall_win_rate': sum(all_win_rates) / len(all_win_rates),
        'win_rate_range': [min(all_win_rates), max(all_win_rates)],
        'consistent_performance': len([wr for wr in all_win_rates if wr > 30]) / len(all_win_rates)
    }
    
    # Save summary
    summary_file = f"results/poker_research_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print key findings
    print(f"\n=== KEY RESEARCH FINDINGS ===")
    print(f"Total successful experiments: {len(successful_results)}")
    
    if 'gpu_speedup' in summary['performance_metrics']:
        speedup = summary['performance_metrics']['gpu_speedup']
        print(f"GPU acceleration: {speedup:.2f}x faster than CPU")
    
    overall_wr = summary['sum_effectiveness']['overall_win_rate']
    print(f"SUM agent overall win rate: {overall_wr:.1f}%")
    
    consistency = summary['sum_effectiveness']['consistent_performance']
    print(f"Performance consistency: {consistency:.1%} of experiments > 30% win rate")
    
    print(f"\nDetailed analysis saved to: {summary_file}")

def run_quick_validation():
    """Quick validation of SUM poker implementation"""
    print("\n=== QUICK SUM POKER VALIDATION ===")
    
    configs = [
        {'name': 'Quick_CPU_Test', 'device': 'cpu', 'hands': 50, 'opponents': 'standard'},
        {'name': 'Quick_GPU_Test', 'device': 'cuda', 'hands': 50, 'opponents': 'standard'}
    ]
    
    results = []
    for config in configs:
        result = run_poker_experiment(config)
        results.append(result)
        
        if result['success']:
            print(f"{config['name']}: WIN RATE {result['win_rate']:.1f}% "
                  f"({result['hands_per_second']:.1f} hands/s)")
        else:
            print(f"{config['name']}: FAILED")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='SUM Poker Research Framework')
    parser.add_argument('--mode', choices=['validation', 'comprehensive', 'custom'], 
                       default='validation', help='Research mode')
    parser.add_argument('--hands', type=int, default=100, 
                       help='Number of hands for custom mode')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda',
                       help='Device for custom mode')
    
    args = parser.parse_args()
    
    print("Strategic Uncertainty Management (SUM) Poker Research")
    print("=" * 55)
    print(f"Mode: {args.mode}")
    
    if args.mode == 'validation':
        results = run_quick_validation()
        print("\nValidation complete. Use --mode comprehensive for full study.")
    
    elif args.mode == 'comprehensive':
        print("\nStarting comprehensive research study...")
        print("This will run 9 experiments and may take 30-60 minutes.")
        confirm = input("Continue? (y/N): ")
        if confirm.lower() == 'y':
            results = run_comprehensive_study()
        else:
            print("Study cancelled.")
            return
    
    elif args.mode == 'custom':
        config = {
            'name': f'Custom_{args.device}_{args.hands}',
            'device': args.device,
            'hands': args.hands,
            'opponents': 'standard'
        }
        result = run_poker_experiment(config)
        if result['success']:
            print(f"\nCustom experiment: {result['win_rate']:.1f}% win rate")
        else:
            print(f"\nCustom experiment failed: {result['output'][:200]}")
    
    print("\nPoker research framework ready for publication-quality experiments.")

if __name__ == '__main__':
    main()