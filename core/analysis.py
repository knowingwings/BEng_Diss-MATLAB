# core/analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

def analyze_results(results_file):
    """Analyze experimental results and generate statistical analysis
    
    Args:
        results_file: Path to results CSV file
        
    Returns:
        dict: Analysis results
    """
    print(f"Analyzing results from {results_file}...")
    
    # Load results
    results = pd.read_csv(results_file)
    
    # Check if this is a stats file or detailed results
    if 'run_index' in results.columns:
        # Detailed results - aggregate first
        grouped = results.groupby(['num_tasks', 'comm_delay', 'packet_loss', 'epsilon'])
        results = grouped.mean().reset_index()
    
    # ANOVA analysis for each response variable
    response_vars = ['makespan', 'message_count', 'optimality_gap', 'recovery_time']
    factor_vars = ['num_tasks', 'comm_delay', 'packet_loss', 'epsilon']
    
    anova_results = {}
    
    for response in response_vars:
        if response not in results.columns:
            continue
            
        print(f"\nANOVA for {response}")
        formula = f"{response} ~ C(num_tasks) + C(comm_delay) + C(packet_loss) + C(epsilon)"
        model = ols(formula, data=results).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)
        
        anova_results[response] = anova_table
        
        # Effect sizes
        print("\nEffect sizes (partial eta-squared):")
        ss_total = anova_table['sum_sq'].sum()
        for factor in factor_vars:
            factor_name = f"C({factor})"
            if factor_name in anova_table.index:
                ss_factor = anova_table.loc[factor_name, 'sum_sq']
                eta_squared = ss_factor / ss_total
                print(f"  {factor}: {eta_squared:.4f}")
    
    # Regression analysis for parameter sensitivity
    print("\nRegression Analysis")
    regression_results = {}
    
    for response in response_vars:
        if response not in results.columns:
            continue
            
        X = results[factor_vars]
        X = sm.add_constant(X)
        y = results[response]
        
        model = sm.OLS(y, X).fit()
        print(f"\nRegression for {response}")
        print(model.summary())
        
        regression_results[response] = {
            'params': model.params,
            'pvalues': model.pvalues,
            'rsquared': model.rsquared,
            'rsquared_adj': model.rsquared_adj
        }
    
    # Calculate confidence intervals
    print("\n95% Confidence Intervals")
    confidence_intervals = {}
    
    for response in response_vars:
        if response not in results.columns:
            continue
            
        mean = results[response].mean()
        ci = stats.t.interval(0.95, len(results)-1, loc=mean, scale=stats.sem(results[response]))
        print(f"{response}: {mean:.2f} ({ci[0]:.2f}, {ci[1]:.2f})")
        
        confidence_intervals[response] = {
            'mean': mean,
            'lower': ci[0],
            'upper': ci[1]
        }
    
    # Return results for potential further processing
    analysis_results = {
        'anova': anova_results,
        'regression': regression_results,
        'confidence_intervals': confidence_intervals,
        'summary_stats': results.describe()
    }
    
    # Save analysis results
    output_dir = os.path.dirname(results_file)
    analysis_file = os.path.join(output_dir, 'analysis_results.pkl')
    import pickle
    with open(analysis_file, 'wb') as f:
        pickle.dump(analysis_results, f)
    
    print(f"Analysis results saved to {analysis_file}")
    
    return analysis_results

def generate_reports(results_file, output_dir=None):
    """Generate plots and visualization of experimental results
    
    Args:
        results_file: Path to results CSV file
        output_dir: Output directory for plots (defaults to results_file directory)
    """
    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    
    print(f"Generating reports from {results_file}...")
    
    # Load results
    results = pd.read_csv(results_file)
    
    # Check if this is a stats file or detailed results
    is_detailed = 'run_index' in results.columns
    
    # If detailed, we can also do analysis by run
    if is_detailed:
        # We'll use both the detailed data and aggregated data
        grouped = results.groupby(['num_tasks', 'comm_delay', 'packet_loss', 'epsilon'])
        agg_results = grouped.mean().reset_index()
    else:
        agg_results = results
    
    # Factor variables and response variables
    factor_vars = ['num_tasks', 'comm_delay', 'packet_loss', 'epsilon']
    response_vars = ['makespan', 'message_count', 'optimality_gap', 'recovery_time']
    response_vars = [v for v in response_vars if v in results.columns]
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Main effects plots
    for response in response_vars:
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Main Effects on {response}', fontsize=16)
        axs = axs.flatten()
        
        for i, factor in enumerate(factor_vars):
            if is_detailed:
                # With detailed data, we can calculate confidence intervals
                means = agg_results.groupby(factor)[response].mean().reset_index()
                ci = results.groupby(factor)[response].sem().reset_index()
                ci[response] = ci[response] * 1.96  # 95% CI
                
                axs[i].errorbar(means[factor], means[response], yerr=ci[response], 
                               marker='o', capsize=5, linewidth=2, markersize=8)
            else:
                # With aggregated data, just plot the means
                means = agg_results.groupby(factor)[response].mean().reset_index()
                axs[i].plot(means[factor], means[response], 
                           marker='o', linewidth=2, markersize=8)
                
            axs[i].set_xlabel(factor, fontsize=12)
            axs[i].set_ylabel(response, fontsize=12)
            axs[i].set_title(f'Main effect of {factor}', fontsize=14)
            axs[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'main_effects_{response}.png'), dpi=300)
        plt.close()
    
    # 2. Interaction plots
    for response in response_vars:
        for i, factor1 in enumerate(factor_vars):
            for j, factor2 in enumerate(factor_vars[i+1:], i+1):
                plt.figure(figsize=(10, 8))
                
                # Create interaction plot
                interaction_data = agg_results.pivot_table(
                    values=response, 
                    index=factor1, 
                    columns=factor2,
                    aggfunc='mean'
                )
                
                # Plot as heatmap
                sns.heatmap(interaction_data, annot=True, cmap='viridis', fmt='.2f', 
                           linewidths=.5, cbar_kws={'label': response})
                
                plt.title(f'Interaction Effect of {factor1} and {factor2} on {response}', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'interaction_{response}_{factor1}_{factor2}.png'), dpi=300)
                plt.close()
    
    # 3. Response distributions
    if is_detailed:
        plt.figure(figsize=(15, 10))
        for i, response in enumerate(response_vars, 1):
            plt.subplot(2, 2, i)
            sns.histplot(results[response], kde=True, bins=20)
            plt.title(f'Distribution of {response}', fontsize=14)
            plt.xlabel(response, fontsize=12)
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'response_distributions.png'), dpi=300)
        plt.close()
    
    # 4. Correlation matrix of response variables
    plt.figure(figsize=(10, 8))
    corr_matrix = agg_results[response_vars].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
               linewidths=.5, square=True)
    plt.title('Correlation Matrix of Response Variables', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'), dpi=300)
    plt.close()
    
    # 5. Parameter sensitivity plot (regression coefficients)
    # Run regression for each response
    plt.figure(figsize=(12, 10))
    
    for i, response in enumerate(response_vars, 1):
        plt.subplot(2, 2, i)
        
        # Fit regression model
        X = sm.add_constant(agg_results[factor_vars])
        y = agg_results[response]
        model = sm.OLS(y, X).fit()
        
        # Extract standardized coefficients
        coefs = model.params[1:]  # Skip intercept
        stds = np.array([agg_results[f].std() for f in factor_vars])
        y_std = agg_results[response].std()
        std_coefs = coefs * stds / y_std
        
        # Plot coefficients
        bars = plt.bar(factor_vars, std_coefs)
        for bar, pval in zip(bars, model.pvalues[1:]):
            if pval < 0.05:
                bar.set_color('green')
            elif pval < 0.1:
                bar.set_color('yellow')
            else:
                bar.set_color('gray')
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title(f'Standardized Coefficients for {response}', fontsize=14)
        plt.xlabel('Parameters', fontsize=12)
        plt.ylabel('Standardized Coefficient', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'parameter_sensitivity.png'), dpi=300)
    plt.close()
    
    # 6. Generate optimality gap vs communication parameters
    if 'optimality_gap' in response_vars:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Optimality gap vs comm_delay
        sns.boxplot(x='comm_delay', y='optimality_gap', data=agg_results, ax=ax1)
        ax1.set_title('Optimality Gap vs Communication Delay', fontsize=14)
        ax1.set_xlabel('Communication Delay (ms)', fontsize=12)
        ax1.set_ylabel('Optimality Gap', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Optimality gap vs packet_loss
        sns.boxplot(x='packet_loss', y='optimality_gap', data=agg_results, ax=ax2)
        ax2.set_title('Optimality Gap vs Packet Loss Probability', fontsize=14)
        ax2.set_xlabel('Packet Loss Probability', fontsize=12)
        ax2.set_ylabel('Optimality Gap', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'optimality_gap_comm_params.png'), dpi=300)
        plt.close()
    
    # 7. Performance across epsilon values
    plt.figure(figsize=(12, 8))
    for response in response_vars:
        # Normalize response for better comparison
        norm_response = agg_results.groupby('epsilon')[response].mean()
        norm_response = norm_response / norm_response.max()
        
        plt.plot(agg_results['epsilon'].unique(), norm_response, 
                marker='o', linewidth=2, label=response)
    
    plt.xlabel('Epsilon (minimum bid increment)', fontsize=12)
    plt.ylabel('Normalized Performance', fontsize=12)
    plt.title('Effect of Epsilon on Performance Metrics', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'epsilon_performance.png'), dpi=300)
    plt.close()
    
    print(f"Reports generated in {plots_dir}")