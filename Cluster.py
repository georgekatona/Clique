class Cluster:
    def __init__(self, dense_units, dimensions, data_points):
        self.dense_units = dense_units
        self.dimensions = dimensions
        self.data_points = data_points

    def __str__(self):
        dense_units_string = str(self.dense_units.tolist())
        data_points_string = ""
        for point in self.data_points.tolist():
            data_points_string = data_points_string + str(point) + "\n"

        return "Dense units: " + dense_units_string + "\nDimensions: " \
               + (str(self.dimensions)) + "\nCluster size: " + (str(len(self.data_points))) \
               + "\nData points:\n" + data_points_string
