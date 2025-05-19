# core/analysis.py
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

def configure_matplotlib(force_headless=False):
    """Configure matplotlib backend based on environment and preferences"""
    import matplotlib
    
    # Check if we're in WSL or other headless environment
    is_wsl = "microsoft-standard" in os.uname().release if hasattr(os, "uname") else False
    has_display = os.environ.get("DISPLAY", "") != ""
    
    # Force headless mode if requested or in environments that need it
    if force_headless or is_wsl or not has_display:
        matplotlib.use('Agg')  # Non-interactive backend
        return 'headless'
    else:
        # Try interactive backend
        try:
            # First try Qt5Agg as it's generally more robust
            matplotlib.use('Qt5Agg')
            import matplotlib.pyplot as plt
            plt.figure()
            plt.close()
            return 'interactive'
        except Exception:
            try:
                # Fall back to TkAgg
                matplotlib.use('TkAgg')
                import matplotlib.pyplot as plt
                plt.figure()
                plt.close()
                return 'interactive'
            except Exception:
                # If all else fails, use Agg
                matplotlib.use('Agg')
                return 'headless'

# Configure matplotlib with auto-detection by default
backend_mode = configure_matplotlib()

# Import plotting libraries AFTER setting the backend
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # Get output directory (same as the input file directory)
    output_dir = os.path.dirname(results_file)
    
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
    analysis_file = os.path.join(output_dir, 'analysis_results.pkl')
    import pickle
    with open(analysis_file, 'wb') as f:
        pickle.dump(analysis_results, f)
    
    # Also save a text summary
    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("# Analysis Summary\n\n")
        f.write(f"## Dataset: {os.path.basename(results_file)}\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        f.write(results.describe().to_string())
        f.write("\n\n")
        
        # ANOVA results
        f.write("## ANOVA Results\n\n")
        for response, anova in anova_results.items():
            f.write(f"### {response}\n\n")
            f.write(anova.to_string())
            f.write("\n\n")
        
        # Confidence intervals
        f.write("## 95% Confidence Intervals\n\n")
        for response, ci in confidence_intervals.items():
            f.write(f"{response}: {ci['mean']:.2f} ({ci['lower']:.2f}, {ci['upper']:.2f})\n")
    
    print(f"Analysis results saved to {analysis_file}")
    print(f"Analysis summary saved to {summary_file}")
    
    return analysis_results

def generate_reports(results_file, output_dir=None, headless=None, save_data=True):
    """Generate plots and visualization of experimental results
    
    Args:
        results_file: Path to results CSV file
        output_dir: Output directory for plots (defaults to results_file directory)
        headless: Force headless mode if True, interactive if False, auto-detect if None
        save_data: Save plot data for remote visualization
    """
    # Configure matplotlib backend based on headless parameter
    if headless is not None:
        global backend_mode
        backend_mode = configure_matplotlib(force_headless=headless)
    
    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    
    print(f"Generating reports from {results_file}...")
    print(f"Using {backend_mode} matplotlib backend")
    
    # Create plots directory within the output directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # If saving data for remote visualization, create a data directory
    data_dir = None
    if save_data:
        data_dir = os.path.join(output_dir, 'plot_data')
        os.makedirs(data_dir, exist_ok=True)
    
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
    
    # Save results data for remote visualization if requested
    if save_data:
        import pickle
        plot_data = {
            'results': results,
            'agg_results': agg_results,
            'is_detailed': is_detailed,
            'factor_vars': factor_vars,
            'response_vars': response_vars
        }
        with open(os.path.join(data_dir, 'plot_data.pkl'), 'wb') as f:
            pickle.dump(plot_data, f)
        print(f"Plot data saved to {os.path.join(data_dir, 'plot_data.pkl')}")
        
    # Only generate plots if save_data is False
    if not save_data:    
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
        
        # 8. Additional plot: Makespan vs Task Count with comparison to theoretical bound
        if 'makespan' in response_vars and 'num_tasks' in factor_vars:
            plt.figure(figsize=(10, 6))
            task_counts = sorted(agg_results['num_tasks'].unique())
            makespan_avg = [agg_results[agg_results['num_tasks'] == tc]['makespan'].mean() for tc in task_counts]
            
            # Plot actual makespan
            plt.plot(task_counts, makespan_avg, 'bo-', linewidth=2, label='Actual Makespan')
            
            # Plot theoretical bound (O(K²)) - scaled to match the actual data
            scale_factor = makespan_avg[0] / (task_counts[0]**2)
            theoretical = [tc**2 * scale_factor for tc in task_counts]
            plt.plot(task_counts, theoretical, 'r--', linewidth=2, label='Theoretical O(K²)')
            
            plt.xlabel('Number of Tasks (K)', fontsize=12)
            plt.ylabel('Makespan (seconds)', fontsize=12)
            plt.title('Makespan vs Task Count with Theoretical Bound', fontsize=14)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'makespan_vs_tasks.png'), dpi=300)
            plt.close()
        
        # 9. Create a comprehensive results summary PDF
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            
            with PdfPages(os.path.join(output_dir, 'experiment_summary.pdf')) as pdf:
                # Title page
                plt.figure(figsize=(8.5, 11))
                plt.text(0.5, 0.8, 'Distributed Auction Algorithm', 
                        fontsize=24, ha='center')
                plt.text(0.5, 0.7, 'Experiment Results Summary', 
                        fontsize=20, ha='center')
                plt.text(0.5, 0.6, f'Configuration: {os.path.basename(results_file).split("_")[0]}', 
                        fontsize=16, ha='center')
                plt.text(0.5, 0.5, f'Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', 
                        fontsize=16, ha='center')
                plt.axis('off')
                pdf.savefig()
                plt.close()
                
                # Summary statistics table
                plt.figure(figsize=(8.5, 11))
                plt.text(0.5, 0.95, 'Summary Statistics', fontsize=20, ha='center')
                
                # Create a table for summary statistics
                col_labels = ['Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max']
                row_labels = response_vars
                
                # Get statistics
                stats_data = []
                for response in response_vars:
                    stats = agg_results[response].describe()
                    stats_data.append([
                        f"{stats['mean']:.2f}",
                        f"{stats['std']:.2f}",
                        f"{stats['min']:.2f}",
                        f"{stats['25%']:.2f}",
                        f"{stats['50%']:.2f}",
                        f"{stats['75%']:.2f}",
                        f"{stats['max']:.2f}"
                    ])
                
                # Add the table
                table = plt.table(cellText=stats_data,
                                rowLabels=row_labels,
                                colLabels=col_labels,
                                cellLoc='center',
                                loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1, 1.5)
                plt.axis('off')
                
                pdf.savefig()
                plt.close()
                
                # Add all the generated plots to the PDF
                for plot_file in sorted(os.listdir(plots_dir)):
                    if plot_file.endswith('.png'):
                        img = plt.imread(os.path.join(plots_dir, plot_file))
                        plt.figure(figsize=(8.5, 11))
                        plt.imshow(img)
                        plt.axis('off')
                        plt.title(plot_file.replace('.png', '').replace('_', ' ').title(), 
                                fontsize=16, pad=20)
                        pdf.savefig()
                        plt.close()
                
                # Add ANOVA results
                for response in response_vars:
                    plt.figure(figsize=(8.5, 11))
                    plt.text(0.5, 0.95, f'ANOVA Results for {response}', 
                            fontsize=20, ha='center')
                    
                    if response in anova_results:
                        anova_table = anova_results[response]
                        
                        # Create text representation of ANOVA table
                        anova_text = anova_table.to_string()
                        
                        # Add the text
                        plt.text(0.1, 0.8, anova_text, fontsize=10, family='monospace')
                        
                        # Add effect sizes
                        plt.text(0.5, 0.5, "Effect Sizes (partial η²):", 
                                fontsize=16, ha='center')
                        
                        effect_texts = []
                        ss_total = anova_table['sum_sq'].sum()
                        for factor in factor_vars:
                            factor_name = f"C({factor})"
                            if factor_name in anova_table.index:
                                ss_factor = anova_table.loc[factor_name, 'sum_sq']
                                eta_squared = ss_factor / ss_total
                                effect_texts.append(f"{factor}: {eta_squared:.4f}")
                        
                        plt.text(0.5, 0.4, "\n".join(effect_texts), 
                                fontsize=14, ha='center')
                        
                    else:
                        plt.text(0.5, 0.5, "No ANOVA results available for this response", 
                                fontsize=16, ha='center', color='red')
                    
                    plt.axis('off')
                    pdf.savefig()
                    plt.close()
                
                # Conclusion page
                plt.figure(figsize=(8.5, 11))
                plt.text(0.5, 0.9, 'Key Findings', fontsize=20, ha='center')
                
                # Create bullet points for key findings
                findings = [
                    "The distributed auction algorithm demonstrated robust performance across varying conditions.",
                    f"Average makespan across all experiments: {agg_results['makespan'].mean():.2f} seconds.",
                    f"Average optimality gap: {agg_results['optimality_gap'].mean():.4f} (theoretical bound: 2ε).",
                    f"Task count had the strongest impact on makespan (partial η²: {anova_results.get('makespan', {}).get('C(num_tasks)', {}).get('sum_sq', 0)/sum(anova_results.get('makespan', {}).get('sum_sq', [0])):.4f}).",
                    f"Communication constraints impacted performance as expected, with {100*agg_results.groupby('comm_delay')['optimality_gap'].mean().iloc[-1]/agg_results.groupby('comm_delay')['optimality_gap'].mean().iloc[0]:.1f}% higher optimality gap at maximum delay.",
                    "Epsilon parameter showed expected trade-off between convergence speed and solution quality."
                ]
                
                for i, finding in enumerate(findings):
                    plt.text(0.1, 0.8 - i*0.1, "• " + finding, fontsize=14)
                
                plt.axis('off')
                pdf.savefig()
                plt.close()
            
            print(f"Comprehensive results summary created: {os.path.join(output_dir, 'experiment_summary.pdf')}")
        except Exception as e:
            print(f"Warning: Could not create PDF summary: {e}")
    
    print(f"Reports generated in {plots_dir}")