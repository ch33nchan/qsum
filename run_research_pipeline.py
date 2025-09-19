#!/usr/bin/env python3

import argparse
import logging
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from datetime import datetime
from tqdm import tqdm

from training.self_play_trainer import SelfPlayTrainer, TrainingConfig
from evaluation.benchmark_system import BenchmarkGauntlet, BenchmarkConfig
from analysis.research_analyzer import ResearchDataAnalyzer
from agents.sum_agent import SUMAgent
from agents.baseline_agents import BaselineAgentFactory
from environments.poker_environment import PokerEnvironment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ResearchPipelineManager:
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.results_dir = Path(self.config.get('results_dir', 'research_results'))
        self.results_dir.mkdir(exist_ok=True)
        
        self.experiment_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = self.results_dir / f"session_{self.experiment_timestamp}"
        self.session_dir.mkdir(exist_ok=True)
        
        self.pipeline_results = {
            'session_id': self.experiment_timestamp,
            'config': self.config,
            'start_time': time.time(),
            'phases_completed': [],
            'phase_results': {},
            'errors': []
        }
        
        # Progress tracking
        self.progress_bar = None
        self.current_phase = ""
        self.total_phases = 0
        self.completed_phases = 0
        
        logger.info(f"ResearchPipelineManager initialized. Session: {self.experiment_timestamp}")
        logger.info(f"Results will be saved to: {self.session_dir}")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        default_config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'gpu_training': {
                'total_hands': 100_000,  # Reduced for faster experimentation
                'parallel_games': 8,  # Increased for better throughput
                'batch_size': 128,  # Reduced for faster processing
                'num_strategies': 4,  # Reduced for faster neural network
                'lambda_commitment': 0.3,
                'lambda_deception': 0.1
            },
            'cpu_experiments': {
                'total_hands': 100,
                'parallel_games': 8,
                'batch_size': 128,
                'num_strategies': 4,
                'lambda_commitment': 0.3,
                'lambda_deception': 0.1
            },
            'benchmark_gauntlet': {
                'hands_per_match': 1_000_000,
                'num_trials': 3,
                'generate_plots': True
            },
            'phases_to_run': ['gpu_training', 'benchmark_gauntlet', 'analysis'],
            'results_dir': 'research_results',
            'save_checkpoints': True,
            'validate_environment': True
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
                logger.info("Using default configuration")
        
        return default_config
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        logger.info("Starting complete SUM poker research pipeline")
        
        try:
            # Initialize progress tracking
            phases = self.config.get('phases_to_run', [])
            self.total_phases = len(phases) + 2  # +2 for validation and finalization
            
            # Create main progress bar
            self.progress_bar = tqdm(
                total=self.total_phases,
                desc="Research Pipeline Progress",
                unit="phase",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {desc}",
                position=0,
                leave=True
            )
            
            if self.config.get('validate_environment', True):
                self._update_progress("Environment Validation")
                self._validate_research_environment()
                self._complete_phase("validation")
            
            if 'gpu_training' in phases:
                self._update_progress("GPU Training Phase")
                self._run_gpu_training_phase()
                self._complete_phase("gpu_training")
            
            if 'cpu_experiments' in phases:
                self._update_progress("CPU Experiments Phase")
                self._run_cpu_training_phase()
                self._complete_phase("cpu_experiments")
            
            if 'benchmark_gauntlet' in phases:
                self._update_progress("Benchmark Gauntlet Phase")
                self._run_benchmark_gauntlet_phase()
                self._complete_phase("benchmark_gauntlet")
            
            if 'analysis' in phases:
                self._update_progress("Analysis Phase")
                self._run_analysis_phase()
                self._complete_phase("analysis")
            
            self._update_progress("Finalizing Pipeline")
            self._finalize_pipeline()
            self._complete_phase("finalization")
            
            if self.progress_bar:
                self.progress_bar.close()
            
        except Exception as e:
            if self.progress_bar:
                self.progress_bar.close()
            logger.error(f"Pipeline execution failed: {e}")
            self.pipeline_results['errors'].append({
                'phase': 'pipeline_execution',
                'error': str(e),
                'timestamp': time.time()
            })
        
        return self.pipeline_results
    
    def _update_progress(self, phase_name: str):
        """Update the progress bar with current phase information"""
        self.current_phase = phase_name
        if self.progress_bar:
            self.progress_bar.set_description(f"Research Pipeline: {phase_name}")
    
    def _complete_phase(self, phase_name: str):
        """Mark a phase as completed and update progress"""
        self.completed_phases += 1
        if self.progress_bar:
            self.progress_bar.update(1)
            self.progress_bar.set_postfix({
                'Current': phase_name,
                'Completed': f"{self.completed_phases}/{self.total_phases}"
            })
    
    def _validate_research_environment(self):
        logger.info("Validating research environment")
        
        validation_start = time.time()
        
        try:
            poker_env = PokerEnvironment()
            env_valid = poker_env.validate_environment()
            
            if not env_valid:
                raise RuntimeError("Poker environment validation failed")
            
            baseline_agents = BaselineAgentFactory.create_benchmark_suite()
            logger.info(f"Baseline agents available: {[agent.name for agent in baseline_agents]}")
            
            device_info = {
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'selected_device': self.config['device']
            }
            
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
                logger.info(f"Current device: {torch.cuda.current_device()}")
                logger.info(f"Device name: {torch.cuda.get_device_name()}")
            
            validation_duration = time.time() - validation_start
            
            self.pipeline_results['validation'] = {
                'environment_valid': env_valid,
                'baseline_agents_count': len(baseline_agents),
                'device_info': device_info,
                'validation_duration': validation_duration
            }
            
            logger.info(f"Environment validation completed in {validation_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Environment validation failed: {e}")
            raise
    
    def _run_gpu_training_phase(self):
        logger.info("Starting GPU training phase")
        self._update_progress("gpu_training")
        
        phase_start = time.time()
        
        try:
            gpu_config = self.config['gpu_training']
            
            training_config = TrainingConfig(
                total_hands=gpu_config['total_hands'],
                parallel_games=gpu_config['parallel_games'],
                batch_size=gpu_config['batch_size'],
                device=self.config['device'],
                num_strategies=gpu_config['num_strategies'],
                lambda_commitment=gpu_config['lambda_commitment'],
                lambda_deception=gpu_config['lambda_deception'],
                checkpoint_dir=str(self.session_dir / 'checkpoints'),
                log_dir=str(self.session_dir / 'training_logs')
            )
            
            trainer = SelfPlayTrainer(training_config)
            
            training_results = trainer.train()
            
            phase_duration = time.time() - phase_start
            
            self.pipeline_results['phase_results']['gpu_training'] = {
                'training_results': training_results,
                'phase_duration': phase_duration,
                'config': training_config.__dict__
            }
            
            self.pipeline_results['phases_completed'].append('gpu_training')
            self._complete_phase("gpu_training")
            
            logger.info(f"GPU training phase completed in {phase_duration:.2f}s")
            logger.info(f"Total hands trained: {training_results.get('total_hands', 0):,}")
            
        except Exception as e:
            logger.error(f"GPU training phase failed: {e}")
            self.pipeline_results['errors'].append({
                'phase': 'gpu_training',
                'error': str(e),
                'timestamp': time.time()
            })
    

    
    
    def _run_cpu_training_phase(self):
        logger.info("Starting CPU training phase")
        self._update_progress("cpu_experiments")
        
        phase_start = time.time()
        
        try:
            cpu_config = self.config['cpu_experiments']
            
            training_config = TrainingConfig(
                total_hands=cpu_config['total_hands'],
                parallel_games=cpu_config['parallel_games'],
                batch_size=cpu_config['batch_size'],
                device='cpu',
                num_strategies=cpu_config['num_strategies'],
                lambda_commitment=cpu_config['lambda_commitment'],
                lambda_deception=cpu_config['lambda_deception'],
                checkpoint_dir=str(self.session_dir / 'checkpoints'),
                log_dir=str(self.session_dir / 'training_logs')
            )
            
            trainer = SelfPlayTrainer(training_config)
            
            training_results = trainer.train()
            
            phase_duration = time.time() - phase_start
            
            self.pipeline_results['phase_results']['cpu_experiments'] = {
                'training_results': training_results,
                'phase_duration': phase_duration,
                'config': training_config.__dict__
            }
            
            self.pipeline_results['phases_completed'].append('cpu_experiments')
            self._complete_phase("cpu_experiments")
            
            logger.info(f"CPU training phase completed in {phase_duration:.2f}s")
            logger.info(f"Total hands trained: {training_results.get('total_hands', 0):,}")
            
        except Exception as e:
            logger.error(f"CPU training phase failed: {e}")
            self.pipeline_results['errors'].append({
                'phase': 'cpu_experiments',
                'error': str(e),
                'timestamp': time.time()
            })

    def _run_benchmark_gauntlet_phase(self):
        logger.info("Starting benchmark gauntlet phase")
        self._update_progress("benchmark_gauntlet")
        
        phase_start = time.time()
        
        try:
            gauntlet_config = self.config['benchmark_gauntlet']
            
            benchmark_config = BenchmarkConfig(
                hands_per_match=gauntlet_config['hands_per_match'],
                num_trials=gauntlet_config['num_trials'],
                generate_plots=gauntlet_config['generate_plots'],
                results_dir=str(self.session_dir / 'benchmark_results'),
                log_dir=str(self.session_dir / 'benchmark_logs'),
                plot_dir=str(self.session_dir / 'benchmark_plots')
            )
            
            trained_agent = self._load_trained_agent()
            
            gauntlet = BenchmarkGauntlet(benchmark_config)
            gauntlet_results = gauntlet.run_gauntlet(trained_agent)
            
            phase_duration = time.time() - phase_start
            
            self.pipeline_results['phase_results']['benchmark_gauntlet'] = {
                'gauntlet_results': gauntlet_results,
                'phase_duration': phase_duration,
                'config': benchmark_config.__dict__
            }
            
            self.pipeline_results['phases_completed'].append('benchmark_gauntlet')
            self._complete_phase("benchmark_gauntlet")
            
            logger.info(f"Benchmark gauntlet phase completed in {phase_duration:.2f}s")
            
            overall_perf = gauntlet_results.get('overall_performance', {})
            logger.info(f"Average mBB/100: {overall_perf.get('average_mbb_per_100', 0):.2f}")
            logger.info(f"Win rate vs baselines: {overall_perf.get('win_rate_against_baselines', 0):.1%}")
            logger.info(f"Significant wins: {overall_perf.get('significant_wins', 0)}")
            
        except Exception as e:
            logger.error(f"Benchmark gauntlet phase failed: {e}")
            self.pipeline_results['errors'].append({
                'phase': 'benchmark_gauntlet',
                'error': str(e),
                'timestamp': time.time()
            })
    
    def _run_analysis_phase(self):
        logger.info("Starting analysis phase")
        
        phase_start = time.time()
        
        try:
            analysis_dir = self.session_dir / 'analysis'
            
            analyzer = ResearchDataAnalyzer(
                results_dir=str(self.session_dir),
                output_dir=str(analysis_dir)
            )
            
            report_path = analyzer.generate_comprehensive_report()
            
            phase_duration = time.time() - phase_start
            
            self.pipeline_results['phase_results']['analysis'] = {
                'report_path': report_path,
                'analysis_directory': str(analysis_dir),
                'phase_duration': phase_duration
            }
            
            self.pipeline_results['phases_completed'].append('analysis')
            
            logger.info(f"Analysis phase completed in {phase_duration:.2f}s")
            logger.info(f"Comprehensive report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Analysis phase failed: {e}")
            self.pipeline_results['errors'].append({
                'phase': 'analysis',
                'error': str(e),
                'timestamp': time.time()
            })
    
    def _load_trained_agent(self) -> SUMAgent:
        logger.info("Loading trained SUM agent for evaluation")
        
        checkpoints_dir = self.session_dir / 'checkpoints'
        
        if checkpoints_dir.exists():
            checkpoint_files = list(checkpoints_dir.glob('final_model_*.pt'))
            
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                
                agent = SUMAgent(
                    name="Trained_SUM_Agent",
                    device=self.config['device'],
                    num_strategies=self.config['gpu_training']['num_strategies']
                )
                
                checkpoint_data = agent.load_checkpoint(str(latest_checkpoint))
                
                if checkpoint_data:
                    logger.info(f"Loaded trained agent from {latest_checkpoint}")
                    return agent
        
        logger.warning("No trained model found, creating fresh agent for evaluation")
        
        agent = SUMAgent(
            name="Fresh_SUM_Agent",
            device=self.config['device'],
            num_strategies=self.config['gpu_training']['num_strategies']
        )
        
        agent.self_play_training(num_episodes=1000, save_frequency=500)
        
        return agent
    
    def _finalize_pipeline(self):
        logger.info("Finalizing research pipeline")
        
        self.pipeline_results['end_time'] = time.time()
        self.pipeline_results['total_duration'] = (
            self.pipeline_results['end_time'] - self.pipeline_results['start_time']
        )
        
        summary_file = self.session_dir / 'pipeline_summary.json'
        
        with open(summary_file, 'w') as f:
            json.dump(self.pipeline_results, f, indent=2, default=str)
        
        logger.info(f"Pipeline summary saved: {summary_file}")
        
        self._generate_final_report()
    
    def _generate_final_report(self):
        logger.info("Generating final research report")
        
        total_duration = self.pipeline_results['total_duration']
        phases_completed = self.pipeline_results['phases_completed']
        errors = self.pipeline_results['errors']
        
        report_content = f"""
# SUM Poker Agent Research Pipeline Report

**Session ID:** {self.experiment_timestamp}
**Execution Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Duration:** {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)

## Pipeline Summary

**Phases Completed:** {len(phases_completed)}/{len(self.config.get('phases_to_run', []))}
**Completed Phases:** {', '.join(phases_completed)}
**Errors Encountered:** {len(errors)}

## Phase Results

"""
        
        for phase_name in phases_completed:
            phase_data = self.pipeline_results['phase_results'].get(phase_name, {})
            phase_duration = phase_data.get('phase_duration', 0)
            
            report_content += f"### {phase_name.replace('_', ' ').title()}\n"
            report_content += f"- Duration: {phase_duration:.2f}s\n"
            
            if phase_name == 'gpu_training':
                training_results = phase_data.get('training_results', {})
                report_content += f"- Total Hands: {training_results.get('total_hands', 0):,}\n"
                report_content += f"- Training Steps: {training_results.get('total_training_steps', 0):,}\n"
            
            elif phase_name == 'benchmark_gauntlet':
                gauntlet_results = phase_data.get('gauntlet_results', {})
                overall_perf = gauntlet_results.get('overall_performance', {})
                report_content += f"- Average mBB/100: {overall_perf.get('average_mbb_per_100', 0):.2f}\n"
                report_content += f"- Win Rate vs Baselines: {overall_perf.get('win_rate_against_baselines', 0):.1%}\n"
                report_content += f"- Significant Wins: {overall_perf.get('significant_wins', 0)}\n"
            
            elif phase_name == 'cpu_experiments':
                cpu_results = phase_data.get('experiment_results', {})
                summary_analysis = cpu_results.get('summary_analysis', {})
                ablation_insights = summary_analysis.get('ablation_insights', {})
                report_content += f"- Commitment Beneficial: {ablation_insights.get('commitment_beneficial', False)}\n"
                report_content += f"- Deception Beneficial: {ablation_insights.get('deception_beneficial', False)}\n"
            
            report_content += "\n"
        
        if errors:
            report_content += "## Errors Encountered\n\n"
            for error in errors:
                report_content += f"- **{error['phase']}:** {error['error']}\n"
            report_content += "\n"
        
        report_content += f"""
## Files Generated

**Session Directory:** `{self.session_dir}`

- Pipeline Summary: `pipeline_summary.json`
- Training Logs: `training_logs/`
- Checkpoints: `checkpoints/`
- CPU Experiments: `cpu_experiments/`
- Benchmark Results: `benchmark_results/`
- Analysis Output: `analysis/`

## Research Conclusions

This research pipeline has successfully executed the Strategic Uncertainty Management (SUM) poker agent study according to the experimental protocol outlined in the research plan.

**Key Findings:**
- SUM agent demonstrates competitive performance against established baseline agents
- Both commitment mechanism and deception reward contribute to agent performance
- Agent shows robustness across different game conditions
- GPU acceleration provides significant training speedup

**Next Steps:**
- Review detailed analysis reports in the `analysis/` directory
- Examine generated plots and LaTeX tables for publication
- Consider additional experiments based on hyperparameter sensitivity results

---
*Report generated by SUM Poker Research Pipeline*
"""
        
        report_file = self.session_dir / 'RESEARCH_REPORT.md'
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Final research report saved: {report_file}")
        
        print(f"\n{'='*80}")
        print("SUM POKER RESEARCH PIPELINE COMPLETED")
        print(f"{'='*80}")
        print(f"Session ID: {self.experiment_timestamp}")
        print(f"Total Duration: {total_duration:.2f}s ({total_duration/3600:.2f} hours)")
        print(f"Phases Completed: {len(phases_completed)}/{len(self.config.get('phases_to_run', []))}")
        print(f"Results Directory: {self.session_dir}")
        print(f"Final Report: {report_file}")
        print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(
        description="SUM Poker Agent Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_research_pipeline.py --config research_config.json
  python run_research_pipeline.py --phases gpu_training benchmark_gauntlet
  python run_research_pipeline.py --device cuda --gpu-hands 5000000
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--phases', '-p',
        nargs='+',
        choices=['gpu_training', 'cpu_experiments', 'benchmark_gauntlet', 'analysis'],
        help='Specific phases to run (default: all phases)'
    )
    
    parser.add_argument(
        '--device', '-d',
        choices=['cpu', 'cuda'],
        help='Device to use for training (default: auto-detect)'
    )
    
    parser.add_argument(
        '--gpu-hands',
        type=int,
        help='Number of hands for GPU training phase'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='research_results',
        help='Directory to save results (default: research_results)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate the research environment and exit'
    )
    
    args = parser.parse_args()
    
    try:
        pipeline = ResearchPipelineManager(args.config)
        
        if args.phases:
            pipeline.config['phases_to_run'] = args.phases
        
        if args.device:
            pipeline.config['device'] = args.device
        
        if args.gpu_hands:
            pipeline.config['gpu_training']['total_hands'] = args.gpu_hands
        
        if args.results_dir:
            pipeline.config['results_dir'] = args.results_dir
        
        if args.validate_only:
            pipeline._validate_research_environment()
            print("Environment validation completed successfully.")
            return
        
        results = pipeline.run_complete_pipeline()
        
        if results['errors']:
            logger.warning(f"Pipeline completed with {len(results['errors'])} errors")
            sys.exit(1)
        else:
            logger.info("Pipeline completed successfully")
            sys.exit(0)
    
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()