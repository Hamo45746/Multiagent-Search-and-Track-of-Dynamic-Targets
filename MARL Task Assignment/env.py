import numpy as np
from pettingzoo.utils.env import AECEnv, ObsType, ActionType, AgentID
#from pettingzoo.utils.conversions import aec_to_parallel_wrapper
import matplotlib as plt
import heapq
from __future__ import annotations
import gymnasium.spaces
from typing import Any, Dict
import agent


class Environment(AECEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'name': "MARLEnvironment"}
    
    def __init__(self, M, N, D, num_agents):
        super().__init__()
        self.M = M  # Grid width
        self.N = N  # Grid height
        self.D = D  # Number of layers

        # Define all agents that may appear in the environment
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents.copy()  # Agents active at any given time

        # Initialize spaces
        self.observation_spaces = {agent: gymnasium.spaces.Box(low=0, high=1, shape=(D, M, N), dtype=np.float32) for agent in self.agents}
        self.action_spaces = {agent: gymnasium.spaces.Discrete(M*N) for agent in self.agents}

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.global_state = np.zeros((D, M, N), dtype=int)
        self.agent_selection = self.agents[0]  # Start with the first agent
    
    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        """
        Reset logic, including resetting the global state and agents' positions.
        """

        self.agents = self.possible_agents.copy()
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.agent_selection = self.agents[0]
    
    def observe(self, agent: AgentID) -> ObsType | None:
        # Assuming each agent tracks its own observations, including history
        observing_agent = self.agents[agent]
        if observing_agent is None:
            return None
    
        # Generate the observation based on the global state and the agent's known state
        new_observation = observing_agent.gen_obs(self.global_state)
        
        # Include updates from other agents within communication range
        #agent.update_obs_from_comms(observing_agent, new_observation)
        
        return new_observation

    def render(self, mode="human") -> None | np.ndarray | str | list:
        fig, ax = plt.subplots()
        for layer in range(self.D):
            for x in range(self.M):
                for y in range(self.N):
                    if self.global_state[layer, x, y] != 0:
                        color = 'blue' if layer == 0 else 'red' if layer == 1 else 'green'
                        ax.scatter(y, x, c=color)  # Note: x and y are inverted for plotting
        plt.xlim(0, self.N)
        plt.ylim(0, self.M)
        plt.grid(True)
        plt.show()
    
    def state(self) -> np.ndarray:
        return self.global_state
    
    def close(self):
        """Closes any resources that should be released.

        Closes the rendering window, subprocesses, network connections,
        or any other resources that should be released.
        """
        pass
    
    def step(self, action: ActionType) -> None:
        """
        Updates agents' positions based on A* pathfinding towards their target locations.
        Handles the action of the current agent_selection in the environment.
        """
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        
        # Assuming action is a tuple (target_x, target_y) for the current agent_selection
        agent_id = self.agent_selection
        target_x, target_y = action
        
        agent_layer_index = self._get_layer_index('agent')
        current_position = np.where(self.global_state[agent_layer_index] == int(agent_id.split('_')[1]))
        
        if current_position[0].size > 0:
            start = (current_position[0][0], current_position[1][0])
            goal = (target_x, target_y)
            
            path = self.a_star_search(start, goal, limit=5)
            
            if len(path) > 1:
                next_step = path[1]
                self.global_state[agent_layer_index, start[0], start[1]] = 0
                self.global_state[agent_layer_index, next_step[0], next_step[1]] = int(agent_id.split('_')[1])
                
                # TODO: Update agent's observation based on new position
                
                # TODO: Implement reward logic based on the action and the outcome
                self.rewards[agent_id] = 0.0  # Placeholder for actual reward calculation
                
                # TODO: Implement termination logic for the agent
                self.terminations[agent_id] = False  # Placeholder for actual termination condition
                self.truncations[agent_id] = False  # Placeholder for actual truncation condition
        
        # Move to the next agent
        next_index = (self.agents.index(self.agent_selection) + 1) % len(self.agents)
        self.agent_selection = self.agents[next_index]
        
        self._accumulate_rewards()  # Accumulate rewards for each agent

    
    def _get_layer_index(self, item_type):
        # Map item types to layer indices
        mapping = {'target': 0, 'jammer': 1, 'agent': 2}
        return mapping.get(item_type, 0)
    
    def a_star_search(self, start, goal, limit=5):
        """
        Executes A* search algorithm from start to goal with a step limit.
        """
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while not len(frontier) == 0:
            current = heapq.heappop(frontier)[1]
            
            if current == goal or len(came_from) - 1 == limit:
                break
            
            for next in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1  # Assuming uniform cost for simplicity
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    heapq.heappush(frontier, (priority, next))
                    came_from[next] = current
        
        return self.reconstruct_path(came_from, start, goal)
    
    def get_neighbors(self, node):
        """
        Returns the neighbors of a given node considering the grid boundaries.
        """
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 4-way movement
        result = []
        for direction in directions:
            next_node = (node[0] + direction[0], node[1] + direction[1])
            if 0 <= next_node[0] < self.M and 0 <= next_node[1] < self.N:
                result.append(next_node)
        return result
    
    def reconstruct_path(self, came_from, start, goal):
        """
        Reconstructs the path from start to goal using the came_from dictionary.
        """
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path
    
    def heuristic(self, a, b):
        """
        Compute the Manhattan distance between two points.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])