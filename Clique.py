import os
import numpy as np


def insertIfJoinCondition(candidates, item, item2, currentDim):
    joined = item + item2

    dims = set()
    for i in range(len(joined)):
        dims.add(int(joined[i][0]))
    if len(dims) == currentDim:
        candidates.append(joined)


def prune(candidates, prev_dim_dense_units):
    for i in range(len(candidates)):
        for j in range(len(candidates[i])):
            if not prev_dim_dense_units.__contains__([candidates[i][j]]):
                print(candidates[i], " pruned")
                candidates.remove(candidates[i])
                break


def self_join(prev_dim_dense_units):
    print("selfjoin")
    candidates = []
    for i in range(len(prev_dim_dense_units)):
        for j in range(i + 1, len(prev_dim_dense_units)):
            insertIfJoinCondition(candidates, prev_dim_dense_units[i], prev_dim_dense_units[j], 2)
    return candidates


# Read in Data with labels (data is normalized, each dimension is in the [0,1] range.

path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "mouse.csv")
X = np.genfromtxt(path, dtype=float, delimiter=' ', usecols=[0, 1])
y = np.genfromtxt(path, dtype="U10", delimiter=' ', usecols=[2])

# 1. Identify Subspaces
xsi = 3
tau = .3

dimension = np.shape(X)[1]
size = np.shape(X)[0]

projection = np.zeros((xsi, dimension))
for d in range(dimension):
    for element in X[:, d]:
        projection[int(element * xsi % xsi), d] += 1

print(projection)
denseUnits = projection > tau * size

print(denseUnits)

oneDimDenseUnits = []

for d in range(dimension):
    for unit in range(xsi):
        if denseUnits[unit, d]:
            oneDimDenseUnits.append([[d, unit]])

print(np.shape(oneDimDenseUnits))

twoDimCandidates = self_join(oneDimDenseUnits)
prune(twoDimCandidates, oneDimDenseUnits)

print(twoDimCandidates)

# Check candidates on data set
# ? Reduce uninteresting subspaces

#


# print(candidates)

# For each dimension k:
#   Generate candidates & prune
#
# return rfn.join_by('dim', denseUnits, denseUnits, jointype='inner', usemask=False)
