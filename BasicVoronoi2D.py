import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.animation import FuncAnimation

# Plot animated Voronoi partitions of dynamic robots in simple 100x100 square environment.

# Environment bounds
x_min, x_max = 0, 100
y_min, y_max = 0, 100

# Set initial position and direction of robots
num_robots = 5
np.random.seed(4)  # For reproducibility
robot_positions = np.random.rand(num_robots, 2) * 100  # Random positions in 100x100 area
robot_directions = np.random.uniform(-1, 1, (num_robots, 2))  # Random initial vector - num_robots x 2 array in range (-1,1)


# Update robot positions - wall bounce dvd style movement
def update_positions(positions, directions):
    for i in range(len(positions)):
        positions[i] += directions[i]  # Move robot per direction
        # Reflect off boundaries
        if positions[i][0] <= x_min or positions[i][0] >= x_max:
            directions[i][0] *= -1
        if positions[i][1] <= y_min or positions[i][1] >= y_max:
            directions[i][1] *= -1
    return positions, directions

# Update frame
def update(frame):
    global robot_positions, robot_directions
    robot_positions, robot_directions = update_positions(robot_positions, robot_directions)
    plt.cla()  # Clear plot
    vor = Voronoi(robot_positions)
    voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=7)
    plt.scatter(robot_positions[:,0], robot_positions[:,1], c='blue', s=100)  # Plot robot positions
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect('equal', adjustable='box')

# Set figure
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=100, interval=200)

# Save animation
# ani.save('Basic_2d_Voronoi_2.gif', writer='pillow', fps=5)

plt.show()
