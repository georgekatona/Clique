class Cluster:
    def __init__(self, dense_units, dimensions, data_point_ids):
        self.id = None
        self.dense_units = dense_units
        self.dimensions = dimensions
        self.data_point_ids = data_point_ids

    def __str__(self):
        return "Dense units: " + str(self.dense_units.tolist()) + "\nDimensions: " \
               + str(self.dimensions) + "\nCluster size: " + str(len(self.data_point_ids)) \
               + "\nData points:\n" + str(self.data_point_ids) + "\n"
