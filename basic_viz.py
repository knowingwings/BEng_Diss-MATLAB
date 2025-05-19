#!/usr/bin/env python3
# basic_viz.py
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def basic_visualize(results_dir, output_dir=None, interactive=False):
    """Create visualizations directly from the CSV results"""
    # Configure matplotlib
    if interactive:
        try:
            import matplotlib
            matplotlib.use('TkAgg')
        except:
            try:
                import matplotlib
                matplotlib.use('Qt5Agg')
            except:
                import matplotlib
                matplotlib.use('Agg')
                print("Warning: Could not set interactive backend")
    
    # Find the CSV files
    potential_files = [
        os.path.join(results_dir, 'experiment_results.csv'),
        os.path.join(results_dir, 'factorial_experiment_stats.csv'),
        os.path.join(results_dir, 'stats_factorial_experiment_detailed.csv'),
        os.path.join(results_dir, 'factorial_experiment_detailed.csv')
    ]
    
    csv_file = None
    for file_path in potential_files:
        if os.path.exists(file_path):
            csv_file = file_path
            break
    
    if csv_file is None:
        # Search for any CSV file
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_file = os.path.join(root, file)
                    break
            if csv_file:
                break
    
    if csv_file is None:
        print(f"Error: No CSV result files found in {results_dir}")
        return
    
    print(f"Using results file: {csv_file}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'visualizations')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    try:
        data = pd.read_csv(csv_file)
        print(f"Loaded data with {len(data)} rows and {len(data.columns)} columns")
        print(f"Column names: {list(data.columns)}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Identify factor variables - more flexible matching
    factor_var_patterns = ['num_tasks', 'comm_delay', 'packet_loss', 'epsilon', 
                          'task', 'delay', 'loss', 'eps']
    factor_vars = []
    
    for col in data.columns:
        col_lower = col.lower()
        for pattern in factor_var_patterns:
            if pattern in col_lower:
                factor_vars.append(col)
                break
    
    # Identify response variables - more flexible matching
    response_var_patterns = ['makespan', 'message_count', 'optimality_gap', 'recovery_time', 'make', 
                           'message', 'count', 'optim', 'gap', 'recovery', '_mean']
    response_vars = []
    
    for col in data.columns:
        col_lower = col.lower()
        for pattern in response_var_patterns:
            if pattern in col_lower and not any(f in col_lower for f in factor_var_patterns):
                response_vars.append(col)
                break
    
    if not factor_vars:
        print("Error: Could not identify factor variables")
        print("Available columns:", list(data.columns))
        return
    
    if not response_vars:
        print("Error: Could not identify response variables")
        print("Available columns:", list(data.columns))
        return
    
    print(f"Factor variables: {factor_vars}")
    print(f"Response variables: {response_vars}")
    
    # Generate visualizations
    
    # 1. Main effects plots
    for response in response_vars:
        if response not in data.columns:
            continue
            
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Main Effects on {response}', fontsize=16)
        axs = axs.flatten()
        
        for i, factor in enumerate(factor_vars):
            if i >= len(axs) or factor not in data.columns:
                continue
                
            means = data.groupby(factor)[response].mean().reset_index()
            axs[i].plot(means[factor], means[response], marker='o', linewidth=2, markersize=8)
            axs[i].set_xlabel(factor, fontsize=12)
            axs[i].set_ylabel(response, fontsize=12)
            axs[i].set_title(f'Main effect of {factor}', fontsize=14)
            axs[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'main_effects_{response}.png'), dpi=300)
        plt.close()
    
    # 2. Correlation matrix
    if len(response_vars) > 1:
        valid_response_vars = [r for r in response_vars if r in data.columns]
        if len(valid_response_vars) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = data[valid_response_vars].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                      linewidths=.5, square=True)
            plt.title('Correlation Matrix of Response Variables', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300)
            plt.close()
    
    # Get task and makespan columns (with flexible names)
    task_col = next((col for col in factor_vars if 'task' in col.lower()), None)
    makespan_col = next((col for col in response_vars if 'make' in col.lower()), None)
    
    # 3. Makespan vs Task Count with theoretical bound
    if task_col and makespan_col:
        plt.figure(figsize=(10, 6))
        task_counts = sorted(data[task_col].unique())
        makespan_avg = [data[data[task_col] == tc][makespan_col].mean() for tc in task_counts]
        
        # Plot actual makespan
        plt.plot(task_counts, makespan_avg, 'bo-', linewidth=2, label='Actual Makespan')
        
        # Plot theoretical bound (O(K²))
        if task_counts and makespan_avg and task_counts[0] > 0 and makespan_avg[0] > 0:
            scale_factor = makespan_avg[0] / (task_counts[0]**2)
            theoretical = [tc**2 * scale_factor for tc in task_counts]
            plt.plot(task_counts, theoretical, 'r--', linewidth=2, label='Theoretical O(K²)')
        
        plt.xlabel('Number of Tasks (K)', fontsize=12)
        plt.ylabel('Makespan (seconds)', fontsize=12)
        plt.title('Makespan vs Task Count with Theoretical Bound', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'makespan_vs_tasks.png'), dpi=300)
        plt.close()
    
    # Get optimality gap, comm delay and packet loss columns (with flexible names)
    gap_col = next((col for col in response_vars if 'gap' in col.lower() or 'optim' in col.lower()), None)
    delay_col = next((col for col in factor_vars if 'delay' in col.lower()), None)
    loss_col = next((col for col in factor_vars if 'loss' in col.lower() or 'packet' in col.lower()), None)
    
    # 4. Optimality gap vs Communication parameters
    if gap_col and delay_col and loss_col:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Optimality gap vs comm_delay
        sns.boxplot(x=delay_col, y=gap_col, data=data, ax=ax1)
        ax1.set_title('Optimality Gap vs Communication Delay', fontsize=14)
        ax1.set_xlabel('Communication Delay (ms)', fontsize=12)
        ax1.set_ylabel('Optimality Gap', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Optimality gap vs packet_loss
        sns.boxplot(x=loss_col, y=gap_col, data=data, ax=ax2)
        ax2.set_title('Optimality Gap vs Packet Loss Probability', fontsize=14)
        ax2.set_xlabel('Packet Loss Probability', fontsize=12)
        ax2.set_ylabel('Optimality Gap', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimality_gap_comm_params.png'), dpi=300)
        plt.close()
    
    # Get epsilon column (with flexible name)
    eps_col = next((col for col in factor_vars if 'eps' in col.lower()), None)
    
    # 5. Performance across epsilon values
    if eps_col:
        plt.figure(figsize=(12, 8))
        
        valid_responses = [r for r in response_vars if r in data.columns]
        for response in valid_responses:
            # Normalize response for better comparison
            eps_values = sorted(data[eps_col].unique())
            response_means = []
            
            for eps in eps_values:
                filtered = data[data[eps_col] == eps]
                response_means.append(filtered[response].mean())
            
            # Normalize
            max_value = max(response_means) if response_means else 1
            if max_value > 0:
                norm_response = [val/max_value for val in response_means]
                plt.plot(eps_values, norm_response, marker='o', linewidth=2, label=response)
        
        plt.xlabel('Epsilon (minimum bid increment)', fontsize=12)
        plt.ylabel('Normalized Performance', fontsize=12)
        plt.title('Effect of Epsilon on Performance Metrics', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'epsilon_performance.png'), dpi=300)
        plt.close()
    
    print(f"Visualizations generated in {output_dir}")
    
    # Show interactive visualization if requested
    if interactive:
        plt.figure(figsize=(10, 8))
        plt.title("Your Experiment Results")
        
        if task_col and makespan_col:
            task_counts = sorted(data[task_col].unique())
            makespan_avg = [data[data[task_col] == tc][makespan_col].mean() for tc in task_counts]
            
            plt.plot(task_counts, makespan_avg, 'bo-', linewidth=2, label='Actual Makespan')
            
            if task_counts and makespan_avg and task_counts[0] > 0 and makespan_avg[0] > 0:
                scale_factor = makespan_avg[0] / (task_counts[0]**2)
                theoretical = [tc**2 * scale_factor for tc in task_counts]
                plt.plot(task_counts, theoretical, 'r--', linewidth=2, label='Theoretical O(K²)')
            
            plt.xlabel('Number of Tasks (K)', fontsize=12)
            plt.ylabel('Makespan (seconds)', fontsize=12)
            plt.title('Makespan vs Task Count with Theoretical Bound', fontsize=14)
            plt.legend()
            plt.grid(True)
        
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display interactive plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Basic visualization of experiment results')
    parser.add_argument('results_dir', help='Directory containing experiment results')
    parser.add_argument('--output', help='Output directory for visualizations')
    parser.add_argument('--interactive', action='store_true', help='Show interactive plots')
    args = parser.parse_args()
    
    basic_visualize(args.results_dir, args.output, args.interactive)