import os
import sys

import numpy as np
import scipy.sparse.csgraph

from Cluster import Cluster


# Inserts joined item into candidates list only if its dimensionality fits
def insert_if_join_condition(candidates, item, item2, current_dim):
    joined = item + item2

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
            insert_if_join_condition(candidates, prev_dim_dense_units[i], prev_dim_dense_units[j], dim)
    return candidates


def is_data_in_projection(tuple, candidate, xsi):
    # print("tuple: ", tuple)
    # print("candidate: ", candidate)
    for dim in candidate:
        element = tuple[dim[0]]
        if int(element * xsi % xsi) != dim[1]:
            return False
    return True


def get_dense_units_for_dim(data, prev_dim_dense_units, dim, xsi, tau):
    candidates = self_join(prev_dim_dense_units, dim)
    prune(candidates, prev_dim_dense_units)
    print("candidates: ", candidates)

    # Count number of elements in candidates
    projection = np.zeros(len(candidates))
    number_of_data_points = np.shape(data)[0]
    for dataIndex in range(number_of_data_points):
        for i in range(len(candidates)):
            if is_data_in_projection(data[dataIndex], candidates[i], xsi):
                projection[i] += 1
    print("projection: ", projection)
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
    file = open(os.path.join(os.path.abspath(os.path.dirname(__file__)), file_name), encoding='utf-8', mode="w+")
    for i, c in enumerate(clusters):
        file.write("Cluster " + str(i) + ":\n" + str(c))
    file.close()


def get_cluster_data_points(data, cluster_dense_units, xsi):
    cluster_points = []

    # Loop through all dense unit
    for i in range(np.shape(cluster_dense_units)[0]):
        tmp_points = data
        # Loop through all dimensions of dense unit
        for j in range(np.shape(cluster_dense_units)[1]):
            feature_index = cluster_dense_units[i][j][0]
            range_index = cluster_dense_units[i][j][1]
            tmp_points = tmp_points[np.where(np.floor(tmp_points[:, feature_index] * xsi % xsi) == range_index)]
        print(tmp_points)
        for element in tmp_points:
            cluster_points.append(element)

    return np.array(cluster_points)


def get_clusters(dense_units, data, xsi):
    graph = build_graph_from_dense_units(dense_units)
    number_of_components, component_list = scipy.sparse.csgraph.connected_components(graph, directed=False)

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
        cluster_data_points = get_cluster_data_points(data, cluster_dense_units, xsi)
        # Add cluster to list
        clusters.append(Cluster(cluster_dense_units, dimensions, cluster_data_points))

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


def normalize_features(data):
    number_of_features = np.shape(data)[1]
    for f in range(number_of_features):
        data[:, f] -= min(data[:, f])
        data[:, f] *= 1 / max(data[:, f])


def run_clique(file_name, feature_columns, label_column, xsi, tau, output_file="clusters.txt"):
    print("Running CLIQUE algorithm on " + file_name + " dataset, feature columns = " +
          str(feature_columns) + ", label column = " + str(label_column) + ", xsi = " +
          str(xsi) + ", tau = " + str(tau) + "\n")

    # Read in data with labels
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), file_name)
    data = np.genfromtxt(path, dtype=float, delimiter=' ', usecols=feature_columns)
    y = np.genfromtxt(path, dtype="U10", delimiter=' ', usecols=[label_column])

    # Normalize each dimension to the [0,1] range
    normalize_features(data)

    # Finding 1 dimensional dense units
    dense_units = get_one_dim_dense_units(data, tau, xsi)

    # Getting 1 dimensional clusters
    clusters = get_clusters(dense_units, data, xsi)

    # Finding dense units and clusters for dimension > 2
    current_dim = 2
    number_of_features = np.shape(data)[1]
    while current_dim <= number_of_features & len(dense_units) > 0:
        print("\n", str(current_dim), " dimensional clusters:")
        dense_units = get_dense_units_for_dim(data, dense_units, current_dim, xsi, tau)
        for cluster in get_clusters(dense_units, data, xsi):
            clusters.append(cluster)
        current_dim += 1

    save_to_file(clusters, output_file)
    print("\nClusters exported to " + output_file)


if __name__ == "__main__":
    # Sample run: python Clique.py mouse.csv [0,1] 2 3 0.3

    if len(sys.argv) > 5:
        run_clique(file_name=sys.argv[1], feature_columns=map(int, sys.argv[2].strip('[]').split(',')),
                   label_column=int(sys.argv[3]), xsi=int(sys.argv[4]), tau=float(sys.argv[5]),
                   output_file=sys.argv[5])
    elif len(sys.argv) > 4:
        run_clique(file_name=sys.argv[1], feature_columns=map(int, sys.argv[2].strip('[]').split(',')),
                   label_column=int(sys.argv[3]), xsi=int(sys.argv[4]), tau=float(sys.argv[5]))
    else:
        # Running with default parameters and data set
        run_clique("mouse.csv", [0, 1], 2, 3, 0.1)
