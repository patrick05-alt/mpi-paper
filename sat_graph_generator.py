import plotly.graph_objects as go
import numpy as np
import os
import json
import argparse
import time
import logging
import sys

# Color codes for terminal output
GREEN = '\033[32m'
YELLOW = '\033[33m'
RED = '\033[31m'
RESET = '\033[0m'

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_benchmark_results(file_path):
    # Load benchmark results from JSON file
    logging.debug(f"Loading benchmark results from {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Benchmark results file not found: {file_path}")
    with open(file_path, "r") as f:
        try:
            results = json.load(f)
            logging.debug(f"Loaded results: {list(results.keys())}")
            return results
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in benchmark results file: {e}")

def generate_plots(results, output_folder):
    # Generate plots using Plotly
    logging.debug(f"Starting plot generation for output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    methods = list(results.keys())
    logging.debug(f"Methods to plot: {methods}")

    # Average Time Bar Chart
    logging.debug("Generating average time bar chart")
    try:
        avg_times = [float(results[m]["avg_time"]) for m in methods]
        logging.debug(f"Average times: {avg_times}")
        fig = go.Figure(data=[go.Bar(x=methods, y=avg_times, marker_color='skyblue')])
        fig.update_layout(
            title='Average Solving Time Comparison',
            yaxis_title='Time (seconds)',
            xaxis_title='Methods',
            showlegend=False
        )
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
        output_path = os.path.join(output_folder, 'average_time.png')
        try:
            start_time = time.time()
            fig.write_image(output_path, engine="kaleido", timeout=60)
            logging.debug(f"Average time plot saved in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logging.warning(f"{YELLOW}Failed to save PNG for average time plot: {e}. Saving as HTML instead{RESET}")
            fig.write_html(os.path.join(output_folder, 'average_time.html'))
    except Exception as e:
        logging.error(f"{RED}Error processing average time data: {e}{RESET}")

    # Average Splits Bar Chart
    logging.debug("Generating average splits bar chart")
    try:
        avg_splits = [float(results[m]["avg_splits"]) for m in methods]
        logging.debug(f"Average splits: {avg_splits}")
        fig = go.Figure(data=[go.Bar(x=methods, y=avg_splits, marker_color='lightgreen')])
        fig.update_layout(
            title='Average Decision Splits Comparison',
            yaxis_title='Number of Splits',
            xaxis_title='Methods',
            showlegend=False
        )
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
        output_path = os.path.join(output_folder, 'average_splits.png')
        try:
            start_time = time.time()
            fig.write_image(output_path, engine="kaleido", timeout=60)
            logging.debug(f"Average splits plot saved in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logging.warning(f"{YELLOW}Failed to save PNG for average splits plot: {e}. Saving as HTML instead{RESET}")
            fig.write_html(os.path.join(output_folder, 'average_splits.html'))
    except Exception as e:
        logging.error(f"{RED}Error processing average splits data: {e}{RESET}")

    # Cactus Plot
    logging.debug("Generating cumulative times cactus plot")
    try:
        fig = go.Figure()
        for method in methods:
            times = sorted([float(t) for t in results[method]["times"]])
            logging.debug(f"Times for {method}: {times[:5]}... ({len(times)} total)")
            if not times:
                logging.warning(f"{YELLOW}No times data for {method}, skipping{RESET}")
                continue
            cumulative_times = np.cumsum(times)
            fig.add_trace(go.Scatter(
                x=list(range(1, len(times) + 1)),
                y=cumulative_times,
                mode='lines',
                name=method
            ))
        fig.update_layout(
            title='Cumulative Solving Times',
            xaxis_title='Number of Instances Solved',
            yaxis_title='Cumulative Time (seconds)',
            legend_title='Methods'
        )
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
        output_path = os.path.join(output_folder, 'cumulative_times.png')
        try:
            start_time = time.time()
            fig.write_image(output_path, engine="kaleido", timeout=60)
            logging.debug(f"Cactus plot saved in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logging.warning(f"{YELLOW}Failed to save PNG for cactus plot: {e}. Saving as HTML instead{RESET}")
            fig.write_html(os.path.join(output_folder, 'cumulative_times.html'))
    except Exception as e:
        logging.error(f"{RED}Error processing cactus plot data: {e}{RESET}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graphs from benchmark results.")
    parser.add_argument("results_file", type=str, help="Path to the benchmark results JSON file.")
    parser.add_argument("--output_folder", type=str, default=".", help="Folder to save the generated plots.")
    args = parser.parse_args()

    try:
        results = load_benchmark_results(args.results_file)
        generate_plots(results, args.output_folder)
        print(f"{GREEN}Generated plots in {args.output_folder} 📊{RESET}")
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        logging.error(f"Script failed: {e}")
        sys.exit(1)