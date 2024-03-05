class State:
    def __init__(self, robot_positions):
        self.robot_positions = robot_positions

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.robot_positions == other.robot_positions

    def __repr__(self):
        return f"State(robot_positions={self.robot_positions})"
