import numpy as np

class Agent:
    def __init__(self, x, y, theta, speed, id, obs_range, comm_range):
        self.x = x
        self.y = y
        self.theta = theta
        self.speed = speed
        self.id = id
        # Initialize the known state space with information about when and where objects were last observed
        self.known_state = np.full((self.D, self.M, self.N), fill_value=-np.inf)
        
    def gen_observation(self, global_state):
        """
        Generate a new observation based on the global state and update the known state.
        """
        observation = np.full((self.D, self.M, self.N), fill_value=-np.inf)
        x, y = self.current_position()
        
        for d in range(len(global_state)):
            for i in range(max(0, x - self.obs_range), min(self.M, x + self.obs_range + 1)):
                for j in range(max(0, y - self.obs_range), min(self.N, y + self.obs_range + 1)):
                    if global_state[d, i, j] != 0:  # Object is currently observed
                        observation[d, i, j] = 0
                        self.known_state[d, i, j] = 0  # Update known state with current observation
                    elif self.known_state[d, i, j] < 0:  # Previously observed location
                        self.known_state[d, i, j] -= 1  # Decrement the observation counter
                        
        return observation
    
    def update_obs_from_comms(self, communicated_states):
        """
        Update the known state with information received from other agents.
        """
        # Merge communicated states into self.known_state considering the timestamp
        pass

    def decide_action(self):
        """
        Decide the next action based on the known state.
        """
        # Decision-making logic here
        pass


    def current_position(self):
        return (self.x, self.y)
