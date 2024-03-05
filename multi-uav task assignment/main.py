import numpy as np
from environment import Environment
from uav import Robot

def main():
    # Environment setup
    env = Environment(0, 100, 0, 100, 10)
    env.generate_nodes()
    env.simulate_obstacle((50, 50), 15)
    
    # Robots setup
    num_robots = 5
    np.random.seed(4) # For reproducibility
    robot_positions = np.random.rand(num_robots, 2) * 100
    robots = [Robot(position) for position in robot_positions]

    # TODO: Implement the logic for robots to move towards equalizing Voronoi partitions and searching within their partition

if __name__ == "__main__":
    main()
