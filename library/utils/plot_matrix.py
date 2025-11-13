from matplotlib import pyplot as plt

def plot_matrix(matrix, title="Matrix", cmap='viridis', figsize=(6,6), vmin=None, vmax=None):
    """
    Plots a given matrix using matplotlib.

    Parameters:
    matrix (np.ndarray): The matrix to be plotted.
    title (str): Title of the plot.
    cmap (str): Colormap to be used for the plot.
    """
    plt.tight_layout()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    plt.show()

    return fig, ax