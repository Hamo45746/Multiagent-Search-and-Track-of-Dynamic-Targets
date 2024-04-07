class Jammer:
    def __init__(self, map_matrix, jam_radius):
        self.map_matrix = map_matrix
        self.position = (0, 0) # Default
        self.radius = jam_radius

    def set_position(self, x, y):
        self.position = (x, y)