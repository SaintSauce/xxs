import matplotlib.pyplot as plt
import pandas as pd

def plot_histogram(
    data: pd.Series,
    bins: int = 30,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Count",
    range: tuple = None
) -> None:
    """ general-purpose histogram with bar annotations """

    plt.figure()
    counts, bin_edges, patches = plt.hist(data.dropna(), bins=bins, range=range)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # annotate each bar with its count
    for count, edge, patch in zip(counts, bin_edges, patches):
        # position at center of bar
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        plt.text(x, y, str(int(count)), ha='center', va='bottom', fontsize='small')

    plt.show()