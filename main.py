# decentralized_control/main.py

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication

from core.simulator import Simulator
from core.experiment_runner import ExperimentRunner
from core.analysis import analyze_results, generate_reports
from gui.visualization import VisualizationApp

def load_config(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Decentralized Control System for Dual Mobile Manipulators')
    parser.add_argument('--mode', choices=['gui', 'experiment', 'analysis'], default='gui',
                       help='Run mode: gui (interactive), experiment (batch), or analysis')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Configuration file path')
    parser.add_argument('--results', type=str, default='results/factorial_experiment.csv',
                       help='Results file path for analysis mode')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of processes for parallel experiment execution')
    args = parser.parse_args()
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    
    if args.mode == 'gui':
        # Run interactive GUI
        app = QApplication(sys.argv)
        config = load_config(args.config)
        window = VisualizationApp(config)
        window.show()
        sys.exit(app.exec_())
    
    elif args.mode == 'experiment':
        # Run factorial experiment
        config = load_config(args.config)
        runner = ExperimentRunner(config)
        results = runner.run_factorial_experiment(num_processes=args.processes)
        runner.save_results(results, args.results)
        
        # Generate basic analysis automatically
        analyze_results(args.results)
        generate_reports(args.results)
    
    elif args.mode == 'analysis':
        # Run analysis on existing results
        if not os.path.exists(args.results):
            print(f"Error: Results file not found: {args.results}")
            return
        
        analysis_results = analyze_results(args.results)
        generate_reports(args.results)
        
        # Return results to allow further analysis in interactive mode
        return analysis_results

if __name__ == '__main__':
    main()