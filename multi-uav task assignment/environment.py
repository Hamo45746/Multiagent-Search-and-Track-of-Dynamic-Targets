import numpy as np
from node import Node
from state import State
from uav import Robot
from scipy.spatial import Voronoi, distance

class Environment:
    LEFT = 'l'
    RIGHT = 'r'
    UP = 'u'
    DOWN = 'd'
    LEFTUP = 'lu'
    LEFTDOWN = 'ld'
    RIGHTUP = 'ru'
    RIGHTDOWN = 'rd'
    
    ACTIONS = {LEFT, RIGHT, UP, DOWN, LEFTUP, LEFTDOWN, RIGHTUP, RIGHTDOWN}
    
    
    def __init__(self, x_min, x_max, y_min, y_max, grid_size):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.grid_size = grid_size
        self.nodes = self.generate_nodes()
        self.robots = [] 
        initial_robot_positions = [] # Initialise with no robots
        self.state = State(initial_robot_positions)
        self.voronoi = None


    def generate_nodes(self):
        nodes = []
        x_positions = np.arange(self.x_min, self.x_max, self.grid_size)
        y_positions = np.arange(self.y_min, self.y_max, self.grid_size)
        for x in x_positions:
            for y in y_positions:
                nodes.append(Node(x, y))
        return nodes


    def create_circular_obstacle(self, center, radius):
        for node in self.nodes:
            if np.sqrt((node.x - center[0])**2 + (node.y - center[1])**2) <= radius:
                node.set_obstacle(True)


    def get_nodes(self):
        return self.nodes
    
    
    def add_robot(self, robot: Robot):
        self.robots += robot
    
    
    def update_robot_positions(self, new_positions):
        self.state = State(new_positions)


    def get_robot_positions(self):
        return self.state.robot_positions
    
    
    # Maybe not needed - have in Robot class
    def perform_action(self, robot_id, action):
        pass
    
    
    def check_valid_position(self, node: Node):
        x, y = node.position()
        # Check bounds
        if x < self.x_min or x >= self.x_max or y < self.y_min or y >= self.y_max:
            return False
        
        # Check obstacle
        if node.is_obstacle:
            return False
        
        return True
    
    
    def update_voronoi(self):
        self.voronoi = Voronoi(self.state.robot_positions)
        
    
    def partition_node_count(self):
        if not self.voronoi:
            self.update_voronoi()
            
        node_positions = np.array([node.position()] for node in self.nodes)

        closest_robot = np.argmin(distance.cdist(node_positions, self.voronoi.points), axis=1)
        partition_nodes = np.bincount(closest_robot, minlength=len(self.voronoi.point_region))
        return partition_nodes
    
    
    def find_target_partition(self, current_index, node_counts, target='less'):
        # Identify the Voronoi region of the current robot
        current_region = self.voronoi.point_region[current_index]
        # Get vertices of current region
        vertices = self.voronoi.regions[current_region]
        
        # Check vertices to find adjacent partitions
        adjacent_regions = []
        for i, region in enumerate(self.voronoi.regions):
            if i == current_region or -1 in region:  # Exclude the current region and outer (unbounded) regions
                continue
            if any(v in vertices for v in region):  # Check for shared vertices
                adjacent_regions.append(i)
        
        # Filter based on target node count
        if target == 'less':
            candidates = [(i, node_counts[i]) for i in adjacent_regions if node_counts[i] < node_counts[current_index]]
        elif target == 'more':
            candidates = [(i, node_counts[i]) for i in adjacent_regions if node_counts[i] > node_counts[current_index]]
        
        # Select partition with max/min node difference
        if candidates:
            if target == 'less':
                chosen_partition = min(candidates, key=lambda x: x[1])
                target_partition_index = chosen_partition[0]
            else:
                chosen_partition = max(candidates, key=lambda x: x[1])
                target_partition_index = chosen_partition[0]
                
            return target_partition_index
        else:
            return current_index  # No suitable adjacent partition found
       
    
    def calculate_target_position(target_partition_index):
        pass
        #return target_position
    
    
    def balance_partitions(self):
        self.update_voronoi()
        node_counts = self.partition_node_count()
        avg_nodes = sum(node_counts) / len(node_counts)
        
        for i, robot in enumerate(self.robots):
            partition_count = node_counts[i]
            if partition_count > avg_nodes:
                # Find adjacent partition with less nodes
                target_partition_index = self.find_target_partition(i, node_counts, target='less')
                target_position = self.calculate_target_position(target_partition_index)
                robot.move_towards(target_position)
            elif partition_count < avg_nodes:
                # Find adjacent partition with more nodes
                target_partition_index = self.find_target_partition(i, node_counts, target='more')
                target_position = self.calculate_target_position(target_partition_index)
                robot.move_towards(target_position)
                
            self.update_robot_positions([robot.current_position() for robot in self.robots])
            self.update_voronoi()
            
    