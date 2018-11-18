import os
import sys

import numpy as np
import scipy.sparse.csgraph

from Cluster import Cluster
from sklearn import metrics
from ast import literal_eval

from Visualization import plot_clusters


# Inserts joined item into candidates list only if its dimensionality fits
def insert_if_join_condition(candidates, item, item2, current_dim):
    joined = []
    for i in range(len(item)):
        joined.append(item[i])
    for i in range(len(item2)):
        joined.append(item2[i])

    # Count number of dimensions
    dims = set()
    for i in range(len(joined)):
        dims.add(int(joined[i][0]))

    # Insert if it fits
    if len(dims) == current_dim:
        candidates.append(joined)


# Prune all candidates, which has a (k-1) dimensional projection not in (k-1) dim dense units
def prune(candidates, prev_dim_dense_units):
    for i in range(len(candidates)):
        for j in range(len(candidates[i])):
            if not prev_dim_dense_units.__contains__([candidates[i][j]]):
                candidates.remove(candidates[i])
                break


def self_join(prev_dim_dense_units, dim):
    candidates = []
    for i in range(len(prev_dim_dense_units)):
        for j in range(i + 1, len(prev_dim_dense_units)):
            insert_if_join_condition(
                candidates, prev_dim_dense_units[i], prev_dim_dense_units[j], dim)
    return candidates


def is_data_in_projection(tuple, candidate, xsi):
    for dim in candidate:
        element = tuple[dim[0]]
        if int(element * xsi % xsi) != dim[1]:
            return False
    return True


def get_dense_units_for_dim(data, prev_dim_dense_units, dim, xsi, tau):
    candidates = self_join(prev_dim_dense_units, dim)
    prune(candidates, prev_dim_dense_units)

    # Count number of elements in candidates
    projection = np.zeros(len(candidates))
    number_of_data_points = np.shape(data)[0]
    for dataIndex in range(number_of_data_points):
        for i in range(len(candidates)):
            if is_data_in_projection(data[dataIndex], candidates[i], xsi):
                projection[i] += 1
    print("projection: ", projection)

    # Return elements above density threshold
    is_dense = projection > tau * number_of_data_points
    print("is_dense: ", is_dense)
    return np.array(candidates)[is_dense]


def build_graph_from_dense_units(dense_units):
    graph = np.identity(len(dense_units))
    for i in range(len(dense_units)):
        for j in range(len(dense_units)):
            graph[i, j] = get_edge(dense_units[i], dense_units[j])
    return graph


def get_edge(node1, node2):
    dim = len(node1)
    distance = 0

    for i in range(dim):
        if node1[i][0] != node2[i][0]:
            return 0
        distance += abs(node1[i][1] - node2[i][1])
        if distance > 1:
            return 0
    return 1


def save_to_file(clusters, file_name):
    file = open(os.path.join(os.path.abspath(os.path.dirname(
        __file__)), file_name), encoding='utf-8', mode="w+")
    for i, c in enumerate(clusters):
        c.id = i
        file.write("Cluster " + str(i) + ":\n" + str(c))
    file.close()


def get_cluster_data_point_ids(data, cluster_dense_units, xsi):
    point_ids = set()

    # Loop through all dense unit
    for i in range(np.shape(cluster_dense_units)[0]):
        tmp_ids = set(range(np.shape(data)[0]))
        # Loop through all dimensions of dense unit
        for j in range(np.shape(cluster_dense_units)[1]):
            feature_index = cluster_dense_units[i][j][0]
            range_index = cluster_dense_units[i][j][1]
            tmp_ids = tmp_ids & set(
                np.where(np.floor(data[:, feature_index] * xsi % xsi) == range_index)[0])
        point_ids = point_ids | tmp_ids

    return point_ids


def get_clusters(dense_units, data, xsi):
    graph = build_graph_from_dense_units(dense_units)
    number_of_components, component_list = scipy.sparse.csgraph.connected_components(
        graph, directed=False)

    dense_units = np.array(dense_units)
    clusters = []
    # For every cluster
    for i in range(number_of_components):
        # Get dense units of the cluster
        cluster_dense_units = dense_units[np.where(component_list == i)]
        print("cluster_dense_units: ", cluster_dense_units.tolist())

        # Get dimensions of the cluster
        dimensions = set()
        for j in range(len(cluster_dense_units)):
            for k in range(len(cluster_dense_units[j])):
                dimensions.add(cluster_dense_units[j][k][0])

        # Get points of the cluster
        cluster_data_point_ids = get_cluster_data_point_ids(
            data, cluster_dense_units, xsi)
        # Add cluster to list
        clusters.append(Cluster(cluster_dense_units,
                                dimensions, cluster_data_point_ids))

    return clusters


def get_one_dim_dense_units(data, tau, xsi):
    number_of_data_points = np.shape(data)[0]
    number_of_features = np.shape(data)[1]
    projection = np.zeros((xsi, number_of_features))
    for f in range(number_of_features):
        for element in data[:, f]:
            projection[int(element * xsi % xsi), f] += 1
    print("1D projection:\n", projection, "\n")
    is_dense = projection > tau * number_of_data_points
    print("is_dense:\n", is_dense)
    one_dim_dense_units = []
    for f in range(number_of_features):
        for unit in range(xsi):
            if is_dense[unit, f]:
                one_dim_dense_units.append([[f, unit]])
    return one_dim_dense_units


# Normalize data in all features (1e-5 padding is added because clustering works on [0,1) interval)
def normalize_features(data):
    normalized_data = data
    number_of_features = np.shape(normalized_data)[1]
    for f in range(number_of_features):
        normalized_data[:, f] -= min(normalized_data[:, f]) - 1e-5
        normalized_data[:, f] *= 1 / (max(normalized_data[:, f]) + 1e-5)
    return normalized_data


def evaluate_clustering_performance(clusters, labels):
    set_of_dimensionality = set()
    for cluster in clusters:
        set_of_dimensionality.add(frozenset(cluster.dimensions))

    # Evaluating performance in all dimensionality
    for dim in set_of_dimensionality:
        print("\nEvaluating clusters in dimension: ", list(dim))
        # Finding clusters with same dimensions
        clusters_in_dim = []
        for c in clusters:
            if c.dimensions == dim:
                clusters_in_dim.append(c)
        clustering_labels = np.zeros(np.shape(labels))
        for i, c in enumerate(clusters_in_dim):
            clustering_labels[list(c.data_point_ids)] = i + 1

        print("Number of clusters: ", len(clusters_in_dim))
        print("Adjusted Rand index: ", metrics.adjusted_rand_score(
            labels, clustering_labels))
        print("Mutual Information: ", metrics.adjusted_mutual_info_score(
            labels, clustering_labels))

        print("Homogeneity, completeness, V-measure: ",
              metrics.homogeneity_completeness_v_measure(labels, clustering_labels))

        print("Fowlkes-Mallows: ",
              metrics.fowlkes_mallows_score(labels, clustering_labels))


def run_clique(data, xsi, tau):
    # Finding 1 dimensional dense units
    dense_units = get_one_dim_dense_units(data, tau, xsi)

    # Getting 1 dimensional clusters
    clusters = get_clusters(dense_units, data, xsi)

    # Finding dense units and clusters for dimension > 2
    current_dim = 2
    number_of_features = np.shape(data)[1]
    while (current_dim <= number_of_features) & (len(dense_units) > 0):
        print("\n", str(current_dim), " dimensional clusters:")
        dense_units = get_dense_units_for_dim(
            data, dense_units, current_dim, xsi, tau)
        for cluster in get_clusters(dense_units, data, xsi):
            clusters.append(cluster)
        current_dim += 1

    return clusters


def read_labels(delimiter, label_column, path):
    return np.genfromtxt(path, dtype="U10", delimiter=delimiter, usecols=[label_column])


def read_data(delimiter, feature_columns, path):
    return np.genfromtxt(path, dtype=float, delimiter=delimiter, usecols=feature_columns)


# Sample run: python Clique.py mouse.csv [0,1] 2 3 0.3 " " output_clusters.txt
if __name__ == "__main__":
    # Clustering with command line parameters
    if len(sys.argv) > 7:
        file_name = sys.argv[1]
        feature_columns = literal_eval(sys.argv[2])
        label_column = int(sys.argv[3])
        xsi = int(sys.argv[4])
        tau = float(sys.argv[5])
        delimiter = sys.argv[6]
        output_file = sys.argv[7]
    # Sample clustering with default parameters
    else:
        file_name = "mouse.csv"
        feature_columns = [0, 1]
        label_column = 2
        xsi = 3
        tau = 0.1
        delimiter = ' '
        output_file = "clusters.txt"

    print("Running CLIQUE algorithm on " + file_name + " dataset, feature columns = " +
          str(feature_columns) + ", label column = " + str(label_column) + ", xsi = " +
          str(xsi) + ", tau = " + str(tau) + "\n")

    # Read in data with labels
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), file_name)
    original_data = read_data(delimiter, feature_columns, path)
    labels = read_labels(delimiter, label_column, path)

    # Normalize each dimension to the [0,1] range
    data = normalize_features(original_data)

    clusters = run_clique(data=data,
                          xsi=xsi,
                          tau=tau)
    save_to_file(clusters, output_file)
    print("\nClusters exported to " + output_file)

    # Evaluate results
    evaluate_clustering_performance(clusters, labels)

    # Visualize clusters
    title = ("DS: " + file_name + " - Params: Tau=" +
             str(tau) + " Xsi=" + str(xsi))
    if len(feature_columns) <= 2:
        plot_clusters(data, clusters, title, xsi)
