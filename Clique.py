import os
import sys

import numpy as np
import scipy.sparse.csgraph


def insert_if_join_condition(candidates, item, item2, current_dim):
    joined = item + item2

    dims = set()
    for i in range(len(joined)):
        dims.add(int(joined[i][0]))
    if len(dims) == current_dim:
        candidates.append(joined)


def prune(candidates, prev_dim_dense_units):
    for i in range(len(candidates)):
        for j in range(len(candidates[i])):
            if not prev_dim_dense_units.__contains__([candidates[i][j]]):
                print(candidates[i], " pruned")
                candidates.remove(candidates[i])
                break


def self_join(prev_dim_dense_units, dim):
    print("\tselfjoin()")
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
    print("get_next_dense_units()")
    print("\tprev dim dense units: ", prev_dim_dense_units)
    number_of_data_points = np.shape(data)[0]
    candidates = self_join(prev_dim_dense_units, dim)
    prune(candidates, prev_dim_dense_units)
    print("\tcandidates: ", candidates)
    projection = np.zeros(len(candidates))
    for dataIndex in range(number_of_data_points):
        for i in range(len(candidates)):
            if is_data_in_projection(data[dataIndex], candidates[i], xsi):
                projection[i] += 1
    print("\tprojection: ", projection)
    is_dense = projection > tau * number_of_data_points
    print("\tis_dense: ", is_dense, "\n")
    return np.array(candidates)[is_dense]


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


def build_graph_from_dense_units(dense_units):
    print("get_graph_from_dense_units()")
    print("\tDense units:\n", dense_units)
    # print("\tShape: ", np.shape(dense_units))
    graph = np.identity(len(dense_units))

    for i in range(len(dense_units)):
        for j in range(len(dense_units)):
            graph[i, j] = get_edge(dense_units[i], dense_units[j])

    print("\tGraph:\n", graph, "\n")
    return graph


def run_clique(file_name, feature_columns, label_column, xsi, tau):
    print("Running CLIQUE algorithm on " + file_name + " dataset, feature columns = " +
          str(feature_columns) + ", label column = " + str(label_column) + ", xsi = " +
          str(xsi) + ", tau = " + str(tau) + "\n")

    # Read in data with labels
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), file_name)
    data = np.genfromtxt(path, dtype=float, delimiter=' ', usecols=feature_columns)
    y = np.genfromtxt(path, dtype="U10", delimiter=' ', usecols=[label_column])

    number_of_features = np.shape(data)[1]
    number_of_data_points = np.shape(data)[0]

    # Normalize each dimension to the [0,1] range
    normalize_features(data, number_of_features)

    # Finding 1 dimensional dense units
    projection = np.zeros((xsi, number_of_features))
    for f in range(number_of_features):
        for element in data[:, f]:
            projection[int(element * xsi % xsi), f] += 1
    print("1D projection:\n", projection, "\n")

    is_dense = projection > tau * number_of_data_points
    print("is_dense:\n", is_dense, "\n")

    one_dim_dense_units = []

    for f in range(number_of_features):
        for unit in range(xsi):
            if is_dense[unit, f]:
                one_dim_dense_units.append([[f, unit]])

    # Getting 1 dimensional clusters
    graph = build_graph_from_dense_units(one_dim_dense_units)
    clusters = scipy.sparse.csgraph.connected_components(graph, directed=False)
    print("Clusters:\n", clusters, "\n")

    # Finding 1 dimensional dense units
    current_dim = 2
    dense_units = get_dense_units_for_dim(data, one_dim_dense_units, 2, xsi, tau)
    # Getting 2 dimensional clusters
    graph = build_graph_from_dense_units(dense_units)
    clusters = scipy.sparse.csgraph.connected_components(graph, directed=False)
    print("Clusters:\n", clusters, "\n")

    # Finding dense units and clusters for dimension > 2
    current_dim += 1
    while current_dim <= number_of_features & len(dense_units) > 0:
        dense_units = get_dense_units_for_dim(data, dense_units, current_dim, xsi, tau)
        graph = build_graph_from_dense_units(dense_units)
        clusters = scipy.sparse.csgraph.connected_components(graph, directed=False)
        print("Clusters:\n", clusters)
        current_dim += 1


def normalize_features(data, number_of_features):
    for f in range(number_of_features):
        data[:, f] -= min(data[:, f])
        data[:, f] *= 1 / max(data[:, f])


if __name__ == "__main__":
    # Sample run: python Clique.py mouse.csv [0,1] 2 3 0.3
    if len(sys.argv) > 4:
        fileName = sys.argv[1]
        feature_columns = map(int, sys.argv[2].strip('[]').split(','))
        label_column = int(sys.argv[3])
        xsi = int(sys.argv[4])
        tau = float(sys.argv[5])
        run_clique(fileName, feature_columns, label_column, xsi, tau)
    else:
        # Running with default parameters and data set
        run_clique("mouse.csv", [0, 1], 2, 3, 0.3)
