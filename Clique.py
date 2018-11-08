import os
import numpy as np
import scipy.sparse.csgraph


def insert_if_join__condition(candidates, item, item2, current_dim):
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
            insert_if_join__condition(candidates, prev_dim_dense_units[i], prev_dim_dense_units[j], dim)
    return candidates


def is_data_in_projection(tuple, candidate):
    # print("tuple: ", tuple)
    # print("candidate: ", candidate)
    for dim in candidate:
        element = tuple[dim[0]]
        if int(element * xsi % xsi) != dim[1]:
            return False
    return True


def get_dense_units_for_dim(prevDimDenseUnits, dim):
    print("get_next_dense_units()")
    print("\tprev dim dense units: ", prevDimDenseUnits)
    candidates = self_join(prevDimDenseUnits, dim)
    prune(candidates, prevDimDenseUnits)
    print("\tcandidates: ", candidates)
    projection = np.zeros(len(candidates))
    for dataIndex in range(dataSize):
        for i in range(len(candidates)):
            if is_data_in_projection(X[dataIndex], candidates[i]):
                projection[i] += 1
    print("\tprojection: ", projection)
    isDense = projection > tau * dataSize
    print("\tisDense: ", isDense)
    return np.array(candidates)[isDense]


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


def get_graph_from_dense_units(denseUnits):
    print("get_graph_from_dense_units()")
    print("\tDense units:\n", denseUnits)
    # print("\tShape: ", np.shape(denseUnits))
    graph = np.identity(len(denseUnits))

    for i in range(len(denseUnits)):
        for j in range(len(denseUnits)):
            graph[i, j] = get_edge(denseUnits[i], denseUnits[j])

    print("\tGraph:\n", graph)
    return graph


# Read in Data with labels (data is normalized, each dimension is in the [0,1] range.
path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "mouse.csv")
X = np.genfromtxt(path, dtype=float, delimiter=' ', usecols=[0, 1])
y = np.genfromtxt(path, dtype="U10", delimiter=' ', usecols=[2])

# Inputs of the algorithm
xsi = 3
tau = .3

# 1. Identify Subspaces
numberOfFeatures = np.shape(X)[1]
dataSize = np.shape(X)[0]

# Finding 1 dimensional dense units
projection = np.zeros((xsi, numberOfFeatures))
for f in range(numberOfFeatures):
    for element in X[:, f]:
        projection[int(element * xsi % xsi), f] += 1
print("1D projection:\n", projection)

isDense = projection > tau * dataSize
print("isDense:\n", isDense)

oneDimDenseUnits = []

for f in range(numberOfFeatures):
    for unit in range(xsi):
        if isDense[unit, f]:
            oneDimDenseUnits.append([[f, unit]])

# Getting 1 dimensional clusters
graph = get_graph_from_dense_units(oneDimDenseUnits)
clusters = scipy.sparse.csgraph.connected_components(graph, directed=False)
print("Clusters:\n", clusters)

# Finding 1 dimensional dense units
currentDim = 2
denseUnits = get_dense_units_for_dim(oneDimDenseUnits, 2)
# Getting 2 dimensional clusters
graph = get_graph_from_dense_units(denseUnits)
clusters = scipy.sparse.csgraph.connected_components(graph, directed=False)
print("Clusters:\n", clusters)

# Finding dense units and clusters for dimension > 2
currentDim += 1
while currentDim <= numberOfFeatures & len(denseUnits) > 0:
    denseUnits = get_dense_units_for_dim(denseUnits, currentDim)
    graph = get_graph_from_dense_units(denseUnits)
    clusters = scipy.sparse.csgraph.connected_components(graph, directed=False)
    print("Clusters:\n", clusters)
    currentDim += 1
