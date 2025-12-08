import matplotlib.pyplot as plt

def plot_sparsity_metric(df, metric, metric_title, model_title, scale=None):
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

    plt.show()