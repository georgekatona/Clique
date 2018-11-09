class Cluster:
    def __init__(self, dense_units, dimensions, data_point_ids):
        self.dense_units = dense_units
        self.dimensions = dimensions
        self.data_point_ids = data_point_ids

    def __str__(self):
        dense_units_string = str(self.dense_units.tolist())

        return "Dense units: " + dense_units_string + "\nDimensions: " \
               + str(self.dimensions) + "\nCluster size: " + str(len(self.data_point_ids)) \
               + "\nData points:\n" + str(self.data_point_ids) + "\n"
