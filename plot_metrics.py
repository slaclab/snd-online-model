import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pytz
import sys

# command: python plot_metrics.py metrics.csv --output metrics_plot.png

def plot_metrics(csv_path, output_path=None):
    """
    Reads a metrics CSV and plots each key as a function of timestamp (human-readable Pacific time date).
    Args:
        csv_path (str): Path to the metrics CSV file.
        output_path (str, optional): Path to save the plot PNG. If None, shows the plot.
    """
    df = pd.read_csv(csv_path)
    # Convert POSIX ms to datetime in UTC, then localize to Pacific Time
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('America/Los_Angeles')
    keys = df['key'].unique()
    plt.figure(figsize=(12, 8))
    for key in keys:
        sub = df[df['key'] == key]
        plt.plot(sub['datetime'], sub['value'], label=key, marker='o', markersize=3, linestyle='-')
    plt.xlabel('Timestamp (Pacific Time)')
    plt.ylabel('Value')
    plt.title('Metrics Over Time')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    # Format x-axis to show readable dates
    ax = plt.gca()
    ax.set_yscale('log')  # Set y-axis to log scale
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M', tz=pytz.timezone('America/Los_Angeles')))
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot metrics from CSV.")
    parser.add_argument("csv_path", help="Path to metrics CSV file.")
    parser.add_argument("--output", help="Path to save plot PNG.")
    args = parser.parse_args()
    plot_metrics(args.csv_path, args.output)

