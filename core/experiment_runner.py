# decentralized_control/core/experiment_runner.py

import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import time
import pickle
import os
import multiprocessing as mp
import random

from core.simulator import Simulator
from core.centralized_solver import CentralizedSolver

class ExperimentRunner:
    def __init__(self, config):
        """Initialize experiment runner
        
        Args:
            config: Experiment configuration 
        """
        self.config = config
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def run_factorial_experiment(self, num_processes=None):
        """Run full factorial experiment based on control variables
        
        Args:
            num_processes: Number of processes for parallel execution (None for serial)
            
        Returns:
            pd.DataFrame: Results dataframe
        """
        params = self.config['parameters']
        
        # Create all parameter combinations
        param_combinations = list(product(
            params['num_tasks'],
            params['comm_delay'],
            params['packet_loss'],
            params['epsilon']
        ))
        
        # Number of repetitions per combination
        num_runs = self.config['experiment'].get('num_runs', 1)
        
        # Expand combinations with run index
        full_combinations = []
        for combo in param_combinations:
            for run in range(num_runs):
                full_combinations.append(combo + (run,))
        
        print(f"Running {len(full_combinations)} experiments "
              f"({len(param_combinations)} parameter combinations x {num_runs} runs)...")
        
        # Set up multiprocessing pool if requested
        if num_processes:
            pool = mp.Pool(processes=num_processes)
            results = list(tqdm(pool.imap(self._run_experiment_config, full_combinations), 
                               total=len(full_combinations)))
            pool.close()
            pool.join()
        else:
            # Serial execution
            results = []
            for combo in tqdm(full_combinations):
                results.append(self._run_experiment_config(combo))
        
        # Combine results into DataFrame
        results_df = pd.DataFrame(results)
        
        # If multiple runs, add statistics
        if num_runs > 1:
            # Group by parameter combinations
            grouped_results = results_df.groupby(['num_tasks', 'comm_delay', 'packet_loss', 'epsilon'])
            
            # Calculate statistics
            stats_df = grouped_results.agg({
                'makespan': ['mean', 'std', 'min', 'max'],
                'message_count': ['mean', 'std', 'min', 'max'],
                'optimality_gap': ['mean', 'std', 'min', 'max'],
                'recovery_time': ['mean', 'std', 'min', 'max'],
                'completion_rate': ['mean', 'std', 'min', 'max'],
                'workload_balance': ['mean', 'std', 'min', 'max']
            }).reset_index()
            
            # Flatten column names
            stats_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                              for col in stats_df.columns.values]
            
            # Save both detailed and statistical results
            self.save_results(results_df, 'factorial_experiment_detailed.csv')
            self.save_results(stats_df, 'factorial_experiment_stats.csv')
            
            return stats_df
        
        return results_df
    
    def _run_experiment_config(self, param_combo):
        """Run experiment with specific parameter configuration
        
        Args:
            param_combo: Parameter combination tuple (num_tasks, comm_delay, packet_loss, epsilon, run_index)
            
        Returns:
            dict: Experiment results
        """
        num_tasks, comm_delay, packet_loss, epsilon, run_index = param_combo
        
        # Set random seed for reproducibility
        random_seed = self.config['experiment']['random_seed'] + run_index
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Initialize simulator
        simulator = Simulator(
            num_robots=self.config['experiment']['robot_count'],
            workspace_size=self.config['experiment']['workspace_size'],
            comm_delay=comm_delay,
            packet_loss=packet_loss,
            epsilon=epsilon
        )
        
        # Generate random tasks
        simulator.generate_random_tasks(num_tasks)
        
        # Get centralized solution for comparison
        centralized_solver = CentralizedSolver()
        optimal_solution = centralized_solver.solve(simulator.robots, simulator.tasks)
        optimal_makespan = optimal_solution['makespan']
        
        # Run simulation with decentralized control
        start_time = time.time()
        simulation_results = simulator.run_simulation(
            self.config['experiment']['simulation_time'],
            inject_failure=self.config['experiment'].get('failure_probability', 0) > 0,
            failure_time_fraction=0.3
        )
        execution_time = time.time() - start_time
        
        # Extract metrics
        makespan = simulation_results['makespan']
        message_count = simulation_results['message_count']
        recovery_time = simulation_results.get('recovery_time', 0)
        
        # Calculate optimality gap
        optimality_gap = (makespan - optimal_makespan) / optimal_makespan if optimal_makespan > 0 else 0
        
        # Create result record
        result = {
            'num_tasks': num_tasks,
            'comm_delay': comm_delay,
            'packet_loss': packet_loss,
            'epsilon': epsilon,
            'run_index': run_index,
            'makespan': makespan,
            'message_count': message_count,
            'optimality_gap': optimality_gap,
            'recovery_time': recovery_time,
            'execution_time': execution_time,
            'completion_rate': simulation_results['completion_rate'],
            'workload_balance': simulation_results['workload_balance'],
            'optimal_makespan': optimal_makespan
        }
        
        return result
    
    def save_results(self, results_df, filename):
        """Save experiment results to CSV
        
        Args:
            results_df: Results DataFrame
            filename: Output filename
        """
        filepath = os.path.join(self.results_dir, filename)
        results_df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        
        # Also save a pickle version for easier reloading
        pickle_path = filepath.replace('.csv', '.pkl')
        results_df.to_pickle(pickle_path)
        print(f"Pickle version saved to {pickle_path}")