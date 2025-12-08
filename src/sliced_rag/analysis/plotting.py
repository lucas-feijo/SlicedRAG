import matplotlib.pyplot as plt

def plot_sparsity_metric(df, metric, metric_title, model_title, scale=None):
    """
    Plot metric vs model sparsity.

    Args:
        df (pandas.DataFrame): Dataframe containing sparsity metrics. Dataframe should be in the format
        used in utils.load_sparsity_eval_results().
        metric (string): Name of dataframe column containing the metric values.
        metric_title (string): Title to be used for the metric axis in the plot.
        model_title (string): Model name to be used in plot title.
        scale (tuple of int or float): minimum and maximum values for the y axis. Default: [0, 1]
    """
    if scale is None:
        scale = [0, 1]

    plt.plot(df["sparsity"], df[metric], marker="o")
    plt.title(f"{metric_title} - {model_title}")
    plt.xlabel("Sparsity")
    plt.ylabel(metric_title)
    plt.legend()
    plt.grid(True)

    ax = plt.gca()
    ax.set_ylim(scale)

    return plt