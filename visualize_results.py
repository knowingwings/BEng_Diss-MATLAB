#!/usr/bin/env python3
# visualize_results.py
import os
import sys
import argparse
import matplotlib
import pickle
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('data_path', type=str, help='Path to plot_data.pkl file or experiment directory')
    parser.add_argument('--interactive', action='store_true', help='Force interactive mode')
    parser.add_argument('--output', type=str, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    # Handle different input patterns
    if os.path.isdir(args.data_path):
        # If directory, look for plot_data.pkl
        data_path = os.path.join(args.data_path, 'plot_data', 'plot_data.pkl')
        if not os.path.exists(data_path):
            # Try alternate location
            data_path = os.path.join(args.data_path, 'plot_data.pkl')
    else:
        # Use provided path directly
        data_path = args.data_path
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find plot data at {data_path}")
        sys.exit(1)
    
    # Configure matplotlib for interactive mode if requested
    if args.interactive:
        try:
            matplotlib.use('Qt5Agg')
        except:
            try:
                matplotlib.use('TkAgg')
            except:
                print("Warning: Could not set interactive backend, using Agg")
                matplotlib.use('Agg')
    else:
        matplotlib.use('Agg')
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(os.path.dirname(data_path), 'plots')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load plot data
    with open(data_path, 'rb') as f:
        plot_data = pickle.load(f)
    
    results = plot_data['results']
    agg_results = plot_data['agg_results']
    is_detailed = plot_data['is_detailed']
    factor_vars = plot_data['factor_vars']
    response_vars = plot_data['response_vars']
    
    print(f"Loaded plot data with {len(response_vars)} response variables")
    print(f"Generating visualizations in {'interactive' if args.interactive else 'headless'} mode")
    
    # Import visualization libraries here to use the configured backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_pdf import PdfPages
    
    # Generate the plots
    # [import the generate_plots function from your core.analysis module]
    from core.analysis import generate_reports
    
    # Call generate_reports with the loaded data
    # This would require modifying generate_reports to accept data directly
    # instead of loading from a file
    
    # Alternatively, create a temporary results file
    tmp_file = os.path.join(output_dir, 'temp_results.csv')
    results.to_csv(tmp_file, index=False)
    
    # Call generate_reports with the appropriate headless setting
    generate_reports(tmp_file, output_dir=output_dir, headless=not args.interactive)
    
    # Clean up temporary file
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    
    print(f"Visualizations generated in {output_dir}")
    
    # If interactive mode, show a sample plot
    if args.interactive:
        plt.figure(figsize=(10, 8))
        plt.title("Interactive Mode Test")
        for response in response_vars:
            norm_response = agg_results.groupby('epsilon')[response].mean()
            norm_response = norm_response / norm_response.max()
            plt.plot(sorted(agg_results['epsilon'].unique()), norm_response, 
                    marker='o', linewidth=2, label=response)
        plt.xlabel('Epsilon (minimum bid increment)', fontsize=12)
        plt.ylabel('Normalized Performance', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()