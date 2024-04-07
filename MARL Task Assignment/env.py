import numpy as np
from pettingzoo.utils.env import AECEnv, ObsType, ActionType, AgentID
from pettingzoo.utils import agent_selector
from pettingzoo.sisl.pursuit.utils import agent_utils, two_d_maps
from pettingzoo.sisl.pursuit.utils.agent_layer import AgentLayer
#from pettingzoo.utils.conversions import aec_to_parallel_wrapper
import matplotlib as plt
import heapq
from __future__ import annotations
import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Any, Dict
import agent
import pygame
import jammer_utils

global OBSTACLE, AGENT, TARGET, JAMMER
TARGET = 1
JAMMER = 2
AGENT = 3
OBSTACLE = 4

class Environment(AECEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'name': "MARLEnvironment"}
    
    def __init__(self, X, Y, D, n_agents, n_targets, n_jammers, obs_range, task_types, max_cycles, pixel_scale=30, render_mode=None):
        """
        Initialise environment.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        super().__init__()
        self.X, self.Y, self.D = X, Y, D
        self.obs_range = obs_range
        self.task_types = task_types
        self._seed()
        # Global state includes layers for map, agents, targets and jammers
        self.map_matrix = two_d_maps.rectangle_map(self.X, self.Y)
        self.global_state = np.zeros((4,) + self.map_matrix.shape, dtype=np.float32)
        
        self.pixel_scale = pixel_scale

        # Create agents
        self.num_agents = n_agents
        self.agents = agent_utils.create_agents(
            self.num_agents, self.map_matrix, self.obs_range, self.np_random
        )
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.agent_layer = AgentLayer(X, Y, self.agents)
        
        # Create targets
        self.num_targets = n_targets
        self.targets = agent_utils.create_agents(
            self.num_targets, self.map_matrix, self.obs_range, self.np_random
        )
        self.target_layer = AgentLayer(X, Y, self.targets)
        
        # Create jammers
        self.num_jammers = n_jammers
        self.jamming_radius = 5
        self.jammers = jammer_utils.create_jammers(
            self.num_jammers, self.map_matrix, self.np_random, self.jamming_radius
        )
         

       
        self.latest_reward_state = [0 for _ in range(self.num_agents)]
        self.latest_done_state = [False for _ in range(self.num_agents)]
        self.latest_obs = [None for _ in range(self.num_agents)]
        
        
        self.agent_selection = agent_selector(self.agents)
        
        self.max_cycles = max_cycles
        
        # Initialize observation and action spaces
        self.observation_spaces = self._init_observation_spaces()
        self.action_spaces = self._init_action_spaces()

        # Reward, termination, and truncation structures
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Pygame for rendering
        self.render_mode = None
        self.screen = None
        
        self.frames = 0
        self.reset()
        
        
    def _init_observation_spaces(self):
        obs_spaces = {}
        for agent in self.agents:
            # Assuming observation includes a portion of the global state around the agent
            obs_spaces[agent] = spaces.Box(low=0, high=1, shape=(self.D, self.obs_range, self.obs_range), dtype=np.float32)
        return obs_spaces
    

    def _init_action_spaces(self):
        # TODO: Correct this - should be entire map grid to allocate locations as tasks
        act_spaces = {}
        for agent in self.agents:
            # Assuming each agent can move in the grid (up, down, left, right) and perform tasks
            act_spaces[agent] = spaces.Discrete(4 + len(self.task_types))  # 4 for movements, rest for tasks
        return act_spaces
    
    
    def _get_layer_index(self, item_type):
        # Map item types to layer indices
        mapping = {'target': TARGET, 'jammer': JAMMER, 'agent': AGENT}
        return mapping.get(item_type, 0)
    
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]


    def action_space(self, agent):
        return self.action_spaces[agent]
    
    
    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        """
        Reset logic, including resetting the global state and agents' positions.
        """
        # Reset global state and other necessary components
        self.global_state.fill(0)
        for agent in self.agents:
            self.rewards[agent] = 0.0
            self.terminations[agent] = False
            self.truncations[agent] = False
            self.infos[agent] = {}
            # Additional reset logic (e.g., randomize agent positions) goes here

        self.agent_selection = agent_selector(self.agents)
        # Note: You'll also want to reset agent positions and tasks here
    
    
    def observe(self, agent: AgentID) -> ObsType | None:
        #TODO: Fix this
        observing_agent = self.agents[agent]
        if observing_agent is None:
            return None
    
        # Generate the observation based on the global state and the agent's known state
        new_observation = observing_agent.gen_observation(self.global_state) # gen_observation not yet implemented
        
        # Include updates from other agents within communication range
        # agent.update_obs_from_comms(observing_agent, new_observation)
        
        return new_observation
    
    
    def draw_model_state(self):
        """
        Use pygame to draw environment map.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        # -1 is building pixel flag - TODO: may need to change this
        x_len, y_len = self.global_state[0].shape
        for x in range(x_len):
            for y in range(y_len):
                pos = pygame.Rect(
                    self.pixel_scale * x,
                    self.pixel_scale * y,
                    self.pixel_scale,
                    self.pixel_scale,
                )
                col = (0, 0, 0)
                if self.global_state[0][x][y] == -1:
                    col = (255, 255, 255)
                pygame.draw.rect(self.screen, col, pos)


    def draw_agents(self):
        """
        Use pygame to draw agents.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        for i in range(self.agent_layer.n_agents()):
            x, y = self.agent_layer.get_position(i)
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            col = (0, 0, 255)
            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3))
            
            
    def draw_targets(self):
        """
        Use pygame to draw targets.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        for i in range(self.target_layer.n_agents()):
            x, y = self.target_layer.get_position(i)
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            col = (255, 0, 0)
            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3))


    def draw_jammers(self):
        """
        Use pygame to draw jammers and jamming regions.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        # Where self.jammers is a list of jammer positions [(x, y), ...] with a uniform jamming radius
        for x, y in self.jammers:  # This loop draws the jammers and their jamming radius
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            # Green for jammers
            col = (0, 255, 0)
            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 4))
            # Draw jamming radius
            jamming_radius_pixels = self.jamming_radius * self.pixel_scale  # Converting jamming radius to pixels
            # Semi-transparent green ellipse for jamming radius
            jamming_area = pygame.Rect(center[0] - jamming_radius_pixels / 2,
                                       center[1] - jamming_radius_pixels / 2,
                                       jamming_radius_pixels,
                                       jamming_radius_pixels
                                    )
            pygame.draw.ellipse(self.screen, (0, 255, 0, 128), jamming_area, width=1)  # Ensure `width` is set to make the ellipse outline


    def draw_agents_observations(self):
        """
        Use pygame to draw agents observation regions.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        for i in range(self.agent_layer.n_agents()):
            x, y = self.agent_layer.get_position(i)
            patch = pygame.Surface(
                (self.pixel_scale * self.obs_range, self.pixel_scale * self.obs_range)
            )
            patch.set_alpha(128)
            patch.fill((72, 152, 255))
            ofst = self.obs_range / 2.0
            self.screen.blit(
                patch,
                (
                    self.pixel_scale * (x - ofst + 1 / 2),
                    self.pixel_scale * (y - ofst + 1 / 2),
                ),
            )
    
    
    def render(self, mode="human") -> None | np.ndarray | str | list:
        """ 
        Basic render of environment using matplotlib scatter plot.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        if self.screen is None:
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.pixel_scale * self.X, self.pixel_scale * self.Y)
                )
                pygame.display.set_caption("Search & Track Task Assign")
            else:
                self.screen = pygame.Surface(
                    (self.pixel_scale * self.X, self.pixel_scale * self.Y)
                )

        self.draw_model_state()
        self.draw_agents_observations()

        self.draw_targets()
        self.draw_agents()

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
        return (
            np.transpose(new_observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )
    
    def state(self) -> np.ndarray:
        return self.global_state
    
    
    def close(self):
        """
        Closes any resources that should be released.

        Closes the rendering window, subprocesses, network connections,
        or any other resources that should be released.
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None
    
    
    def step(self, action: ActionType) -> None:
        """
        Updates agents' positions based on A* pathfinding towards their target locations.
        Handles the action of the current agent_selection in the environment.
        """
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        
        # Action is a tuple (target_x, target_y) for the current agent_selection
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
        
        self._accumulate_rewards()  # Accumulate rewards for each agent - Check implementation for use case
    
    
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
        Returns the neighbors of a given grid node considering the grid boundaries.
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
    
    
    def _seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        # try:
        #     policies = [self.evader_controller, self.pursuer_controller]
        #     for policy in policies:
        #         try:
        #             policy.set_rng(self.np_random)
        #         except AttributeError:
        #             pass
        # except AttributeError:
        #     pass

        return [seed_]