import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# Create plot directory if it doesn't exist
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'plot')
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_histogram(
    data: pd.Series,
    bins: int = 30,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Count",
    range: tuple = None
) -> None:
    """Create a histogram and save it to the plot directory"""
    
    plt.figure(figsize=(10, 6))
    counts, bin_edges, patches = plt.hist(data.dropna(), bins=bins, range=range)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add count annotations
    for count, edge, patch in zip(counts, bin_edges, patches):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        plt.text(x, y, str(int(count)), ha='center', va='bottom', fontsize='small')

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hist_{title.replace(' ', '_')}_{timestamp}.png" if title else f"hist_{timestamp}.png"
    plt.savefig(os.path.join(PLOT_DIR, filename))
    
    plt.show()