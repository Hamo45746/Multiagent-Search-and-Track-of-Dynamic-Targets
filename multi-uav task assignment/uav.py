import numpy as np
import environment
import node

class Robot:
    def __init__(self, position, id):
        self.position = position
        self.id = id

    def move_towards(self, target_position, environment):
        # Find the node one step closer to target
        next_node_position = self.find_next_step(self.position, target_position, environment)
        # Move to the next node position
        self.position = next_node_position

    def find_next_step(self, current_position, target_position, environment):
    
        adjacent_nodes = environment.get_adjacent_nodes(current_position)
        # Calculate which adjacent node is closest to the target position
        closest_node_position = min(adjacent_nodes, key=lambda node: np.linalg.norm(np.array(target_position) - np.array(node.position)))
        return closest_node_position.position

    def current_position(self):
        return self.position
