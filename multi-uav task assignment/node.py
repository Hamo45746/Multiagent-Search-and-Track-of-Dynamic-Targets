class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.is_obstacle = False

    def position(self):
        return (self.x, self.y)
    
    def set_obstacle(self, obstacle=True):
        self.is_obstacle = obstacle
        
