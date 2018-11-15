import math

import numpy as np
import matplotlib.pyplot as plt


def plot_clusters(data, clusters, title, xsi):
    # Check if there are clusters to plot
    if len(clusters) <= 0:
        return

    ndim = data.shape[1]
    nrecords = data.shape[0]
    data_extent = [[min(data[:, x]), max(data[:, x])] for x in range(0, ndim)]
    plt_nrow = math.floor(ndim ** 0.5)
    plt_ncol = plt_nrow * (1 - plt_nrow) + ndim
    plt_cmap = plt.cm.tab10
    plt_marker_size = 10
    plt_spacing = 0  # change spacing to apply a margin to data_extent

    # Plot clusters in each dimension
    for dim in range(1, ndim + 1):
        # Get all clusters in 'dim' dimension(s)
        clusters_in_dim = []
        for c in clusters:
            if len(c.dimensions) == dim:
                clusters_in_dim.append(c)

        # Check if there are clusters in 'dim' dimension(s)
        dim_nclusters = len(clusters_in_dim)
        if dim_nclusters <= 0:
            continue

        # subplot for the current dimension (dim)
        ax = plt.subplot(plt_nrow, plt_ncol, dim)

        # Plot all data points as black points
        if dim == 1:
            ax.scatter(data[:, 0], [0] * nrecords,
                       s=plt_marker_size, c=["black"], label="noise")
            ax.scatter([0] * nrecords, data[:, 1],
                       s=plt_marker_size, c=["black"])
        elif dim == 2:
            ax.scatter(data[:, 0], data[:, 1],
                       s=plt_marker_size, c=["black"], label="noise")

        # For all clusters in 'dim' dimension(s)
        for i, c in enumerate(clusters_in_dim):
            c_size = len(c.data_point_ids)
            c_attrs = list(c.dimensions)
            c_elems = list(c.data_point_ids)

            if dim == 1:  # one-dimensional clusters
                x = data[c_elems, 0] if c_attrs[0] == 0 else [0] * c_size
                y = data[c_elems, 1] if c_attrs[0] == 1 else [0] * c_size
            elif dim == 2:  # two-dimensional clusters
                x = data[c_elems, c_attrs[0]]
                y = data[c_elems, c_attrs[1]]
            ax.scatter(x, y, s=plt_marker_size, c=[
                plt_cmap(c.id)], label=str(c.id))

        ax.set_xlim(data_extent[0][0] - plt_spacing,
                    data_extent[0][1] + plt_spacing)
        ax.set_ylim(data_extent[1][0] - plt_spacing,
                    data_extent[1][1] + plt_spacing)
        ax.set_title(str(dim) + "-dimensional clusters")
        ax.legend(title="Cluster ID")

        # Putting grids on the charts
        minor_ticks_x = np.linspace(
            data_extent[0][0], data_extent[0][1], xsi + 1)
        minor_ticks_y = np.linspace(
            data_extent[1][0], data_extent[1][1], xsi + 1)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(minor_ticks_y, minor=True)
        ax.grid(b=True, which="minor", axis="both")

    plt.gcf().suptitle(title)
    plt.show()
