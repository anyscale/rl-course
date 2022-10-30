"""
Envs in this file:

Module 2:
- MultiAgentArena
- MyCartPole

Module 3:
- FrozenPond
- Maze
- RandomMaze
- RandomLake
- RandomLakeObs
- RandomLakeObsRew
- RandomLakeObsRew2

Module 4 and 5:
- BasicRecommender
- BasicRecommenderWithHistory
"""

import time
import gym
from gym.spaces import Discrete, MultiDiscrete
import numpy as np
# from ipywidgets import Output
from IPython import display

import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import utils

from gym.envs.classic_control import CartPoleEnv
from gym.wrappers import TimeLimit

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


class MyCartPole(TimeLimit):
    def __init__(self, env_config=None):
        if isinstance(env_config, dict):
            env = CartPoleEnv(**env_config)
        else:
            env = CartPoleEnv()
            
        super().__init__(env, max_episode_steps=500)
        
    def render(self):
        utils.my_render_cartpole_matplotlib(self)

def make_cartpole(env_config=None):
    env = MyCartPole(env_config)
    return TimeLimit(env, max_episode_steps=500)
        

# This env created by Sven Mika
# with minor modifications by Mike Gelbart
class MultiAgentArena(MultiAgentEnv):  # MultiAgentEnv is a gym.Env sub-class
    
    def __init__(self, config=None):
        config = config or {}
        # Dimensions of the grid.
        self.width = config.get("width", 6)
        self.height = config.get("height", 6)

        # End an episode after this many timesteps.
        self.timestep_limit = config.get("ts", 50)

        self.observation_space = MultiDiscrete([self.width * self.height,
                                                self.width * self.height])
        # old way: 0=up, 1=right, 2=down, 3=left.
        # frozen lake compatible: 0=left, 1=down, 2=right, 3=up
        self.action_space = Discrete(4)

        # Reset env.
        self.reset()
        
        # For rendering.
        # self.out = None
        # if config.get("render"):
        #     self.out = Output()
        #     display.display(self.out)

        self._spaces_in_preferred_format = False

    def reset(self):
        """Returns initial observation of next(!) episode."""
        # Row-major coords.
        self.agent1_pos = [0, 0]  # upper left corner
        self.agent2_pos = [self.height - 1, self.width - 1]  # lower bottom corner

        # Accumulated rewards in this episode.
        self.agent1_R = 0.0
        self.agent2_R = 0.0

        # Reset agent1's visited fields.
        self.agent1_visited_fields = set([tuple(self.agent1_pos)])

        # How many timesteps have we done in this episode.
        self.timesteps = 0

        # Did we have a collision in recent step?
        self.collision = False
        # How many collisions in total have we had in this episode?
        self.num_collisions = 0

        # Return the initial observation in the new episode.
        return self._get_obs()

    def step(self, action: dict):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.
        
        e.g.
        `action={"agent1": action_for_agent1, "agent2": action_for_agent2}`
        """
        
        # increase our time steps counter by 1.
        self.timesteps += 1
        # An episode is "done" when we reach the time step limit.
        is_done = self.timesteps >= self.timestep_limit

        # Agent2 always moves first.
        # events = [collision|agent1_new_field]
        events = self._move(self.agent2_pos, action["agent2"], is_agent1=False)
        events |= self._move(self.agent1_pos, action["agent1"], is_agent1=True)

        # Useful for rendering.
        self.collision = "collision" in events
        if self.collision is True:
            self.num_collisions += 1
            
        # Get observations (based on new agent positions).
        obs = self._get_obs()

        # Determine rewards based on the collected events:
        r1 = -1.0 if "collision" in events else 1.0 if "agent1_new_field" in events else -0.5
        r2 = 1.0 if "collision" in events else -0.1

        self.agent1_R += r1
        self.agent2_R += r2
        
        rewards = {
            "agent1": r1,
            "agent2": r2,
        }

        # Generate a `done` dict (per-agent and total).
        dones = {
            "agent1": is_done,
            "agent2": is_done,
            # special `__all__` key indicates that the episode is done for all agents.
            "__all__": is_done,
        }

        return obs, rewards, dones, {}  # <- info dict (not needed here).

    def _get_obs(self):
        """
        Returns obs dict (agent name to discrete-pos tuple) using each
        agent's current x/y-positions.
        """
        ag1_discrete_pos = self.agent1_pos[0] * self.width + \
            (self.agent1_pos[1] % self.width)
        ag2_discrete_pos = self.agent2_pos[0] * self.width + \
            (self.agent2_pos[1] % self.width)
        return {
            "agent1": np.array([ag1_discrete_pos, ag2_discrete_pos]),
            "agent2": np.array([ag2_discrete_pos, ag1_discrete_pos]),
        }

    def _move(self, coords, action, is_agent1):
        """
        Moves an agent (agent1 iff is_agent1=True, else agent2) from `coords` (x/y) using the
        given action (0=up, 1=right, etc..) and returns a resulting events dict:
        Agent1: "new" when entering a new field. "bumped" when having been bumped into by agent2.
        Agent2: "bumped" when bumping into agent1 (agent1 then gets -1.0).
        """
        
        # old way: 0=up, 1=right, 2=down, 3=left.
        # frozen lake compatible: 0=left, 1=down, 2=right, 3=up
        ACTION_MAPPING = {
            0 : 3,
            1 : 2,
            2 : 1,
            3 : 0
        }
        action = ACTION_MAPPING[action]
        # above: fix the convention to match frozen lake
        # though Sven's code was originally different
        
        orig_coords = coords[:]
        # Change the row: 0=up (-1), 2=down (+1)
        coords[0] += -1 if action == 0 else 1 if action == 2 else 0
        # Change the column: 1=right (+1), 3=left (-1)
        coords[1] += 1 if action == 1 else -1 if action == 3 else 0

        # Solve collisions.
        # Make sure, we don't end up on the other agent's position.
        # If yes, don't move (we are blocked).
        if (is_agent1 and coords == self.agent2_pos) or (not is_agent1 and coords == self.agent1_pos):
            coords[0], coords[1] = orig_coords
            # Agent2 blocked agent1 (agent1 tried to run into agent2)
            # OR Agent2 bumped into agent1 (agent2 tried to run into agent1)
            return {"collision"}

        # No agent blocking -> check walls.
        if coords[0] < 0:
            coords[0] = 0
        elif coords[0] >= self.height:
            coords[0] = self.height - 1
        if coords[1] < 0:
            coords[1] = 0
        elif coords[1] >= self.width:
            coords[1] = self.width - 1

        # If agent1 -> "new" if new tile covered.
        if is_agent1 and not tuple(coords) in self.agent1_visited_fields:
            self.agent1_visited_fields.add(tuple(coords))
            return {"agent1_new_field"}
        # No new tile for agent1.
        return set()

    def render(self, mode=None):

        # if self.out is not None:
        #     self.out.clear_output(wait=True)
        display.clear_output(wait=True);

        print("_" * (self.width + 2))
        for r in range(self.height):
            print("|", end="")
            for c in range(self.width):
                field = r * self.width + c % self.width
                if self.agent1_pos == [r, c]:
                    print("1", end="")
                elif self.agent2_pos == [r, c]:
                    print("2", end="")
                elif (r, c) in self.agent1_visited_fields:
                    print(".", end="")
                else:
                    print(" ", end="")
            print("|")
        print("‾" * (self.width + 2))
        print(f"{'!!Collision!!' if self.collision else ''}")
        print("R1={: .1f}".format(self.agent1_R))
        print("R2={: .1f} ({} collisions)".format(self.agent2_R, self.num_collisions))
        print(f"Env timesteps={self.timesteps}/{self.timestep_limit}")


        
        
class FrozenPond(gym.Env):
    def __init__(self, env_config=None):
        if env_config is None:
            env_config = dict()
            
        self.size = env_config.get("size", 4)
        self.observation_space = gym.spaces.Discrete(self.size*self.size)
        self.action_space = gym.spaces.Discrete(self.size) 
        
    def reset(self):
        self.player = (0, 0) # the player starts at the top-left
        self.goal = (self.size-1, self.size-1)   # goal is at the bottom-right
        
        if self.size == 4:
            self.holes = np.array([
                [0, 0, 0, 0],  # FFFF
                [0, 1, 0, 1],  # FHFH
                [0, 0, 0, 1],  # FFFH
                [1, 0, 0, 0]  # HFFF
            ])
        else:
            raise Exception("Frozen Pond only supports size 4")
        
        return self.observation()
    
    def observation(self):
        return self.size*self.player[0] + self.player[1]
    
    def reward(self):
        return int(self.player == self.goal)
    
    def done(self):
        return self.player == self.goal or bool(self.holes[self.player] == 1)
        # cast from numpy.bool to bool because of the RLlib check_env
    
    def is_valid_loc(self, location):
        return 0 <= location[0] < self.size and 0 <= location[1] < self.size

    def step(self, action):
        # Compute the new player location
        if action == 0:   # left
            new_loc = (self.player[0], self.player[1]-1)
        elif action == 1: # down
            new_loc = (self.player[0]+1, self.player[1])
        elif action == 2: # right
            new_loc = (self.player[0], self.player[1]+1)
        elif action == 3: # up
            new_loc = (self.player[0]-1, self.player[1])
        else:
            raise ValueError("Action must be in {0,1,2,3}")
        
        # Update the player location only if you stayed in bounds
        # (if you try to move out of bounds, the action does nothing)
        if self.is_valid_loc(new_loc):
            self.player = new_loc
        
        # Return observation/reward/done
        return self.observation(), self.reward(), self.done(), {"player" : self.player, "goal" : self.goal}
    
    def render(self):
        for i in range(self.size):
            for j in range(self.size):
                if (i,j) == self.player:
                    print("🧑", end="")
                elif (i,j) == self.goal:
                    print("⛳️", end="")
                elif self.holes[i,j]:
                    print("🕳", end="")
                else:
                    print("🧊", end="")
            print()


class Maze(FrozenPond):
    def done(self):
        return self.player == self.goal
    def is_valid_loc(self, location):
        return 0 <= location[0] < self.size and 0 <= location[1] < self.size and not self.holes[location]

    def render(self):
        for i in range(self.size):
            for j in range(self.size):
                if (i,j) == self.goal:
                    print("⛳️", end="")
                elif (i,j) == self.player:
                    print("🧑", end="")
                elif self.holes[i,j]:
                    print("🟦", end="")
                else:
                    print("🧊", end="")
                # print("O", end="")
            print()


class RandomMaze(Maze):
    def reset(self):
        self.player = (0, 0) # the player starts at the top-left
        self.goal = (3, 3)   # goal is at the bottom-right
        
        self.walls = np.random.rand(self.size, self.size) < 0.2
        self.walls[self.player] = 0
        self.walls[self.goal] = 0
        
        return self.observation()
    
    def done(self):
        return self.player == self.goal


class RandomLake(FrozenPond):
    def reset(self):
        self.player = (0, 0) # the player starts at the top-left
        self.goal = (self.size-1, self.size-1)   # goal is at the bottom-right
        
        self.holes = np.random.rand(self.size, self.size) < 0.2
        self.holes[self.player] = 0  # no hole at start location
        self.holes[self.goal] = 0    # no hole at goal location
        
        return self.observation()
    
    def seed(self, seed):
        np.random.seed(seed)


class RandomLakeObs(RandomLake):
    def __init__(self, env_config=None):
        if env_config is None:
            env_config = dict()
            
        self.size = env_config.get("size", 4)
        self.observation_space = gym.spaces.MultiDiscrete([2]*self.size)
        self.action_space = gym.spaces.Discrete(self.size)      
    
    def observation(self):
        i, j = self.player

        obs = []
        obs.append(1 if j==0           else self.holes[i,j-1]) # left
        obs.append(1 if i==self.size-1 else self.holes[i+1,j]) # down
        obs.append(1 if j==self.size-1 else self.holes[i,j+1]) # right
        obs.append(1 if i==0           else self.holes[i-1,j]) # up
        
        obs = np.array(obs, dtype=int) # this line is optional, helps readability of output
        return obs


class RandomLakeObsRew(RandomLakeObs): # fails to reach goal, bad reward
    def reward(self):
        return (self.size*2-2)-(abs(self.player[0]-self.goal[0]) + abs(self.player[1]-self.goal[1]))


# class RandomLakeObsTest(RandomLakeObs):
#     def reward(self):
#         goal_rew = int(self.player == self.goal)
#         hole_rew = 0.1*(1-self.holes[self.player])
#         return goal_rew + hole_rew


class RandomLakeObsRew2(RandomLakeObs): # this one should work
    
    # COPIED from above, but with the modification that reward can depend on action
    # not very good code practice, but makes the lesson clearer
    # could also do some sort of wrapper thing
    def step(self, action):
        # Compute the new player location
        if action == 0:   # left
            new_loc = (self.player[0], self.player[1]-1)
        elif action == 1: # down
            new_loc = (self.player[0]+1, self.player[1])
        elif action == 2: # right
            new_loc = (self.player[0], self.player[1]+1)
        elif action == 3: # up
            new_loc = (self.player[0]-1, self.player[1])
        else:
            raise ValueError("Action must be in {0,1,2,3}")
        
        rew = 0
        
        # Update the player location only if you stayed in bounds
        # (if you try to move out of bounds, the action does nothing)
        if self.is_valid_loc(new_loc):
            self.player = new_loc
        else:
            rew -= 0.1
            
        
        # Return observation/reward/done
        return self.observation(), self.reward(action) + rew, self.done(), {"player" : self.player, "goal" : self.goal}
    
    def reward(self, action):
        reward = 0
        if self.holes[self.player]:
            reward -= 0.1
            
        # if action in (0,3):
            # reward -= 0.01
        
        if self.player == self.goal:
            reward += 1
            
        return reward
    
    
    
    
class BasicRecommender(gym.Env):
    def __init__(self, env_config=None):
        
        if env_config is None:
            env_config = dict()
            
        self.num_candidates = env_config.get("num_candidates", 10)
        self.slate_size = env_config.get("slate_size", 1)
        self.resample_documents = env_config.get("resample_documents", True)
        self.max_steps = env_config.get("max_steps", 100)
        self.alpha = env_config.get("alpha", 0.9) # (0,1)
        self.history_len = env_config.get("history_len", 2)  # number of past frames (not including present)
        # Set obser and action space
        self._set_spaces()
        
        # Create the documents
        if "seed" in env_config:
            self.seed(env_config["seed"])
        
    def _set_spaces(self):
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.num_candidates,))
        # if self.slate_size == 1:
        self.action_space = gym.spaces.Discrete(self.num_candidates)
        # else:
            # self.action_space = gym.spaces.MultiDiscrete([self.num_candidates]*self.slate_size) 

    def seed(self, seed):
        np.random.seed(seed)
        
    def reset(self):
        self.sugar_level = 0.0 # starts with no satiety 
        self.step_count = 0
        
        self.history_stack = np.zeros(self.history_len)
        
        self.resample_docs()
        
        return self.observation()
    
    def observation(self):            
        return self.documents # TODO - make this fancier?
    
    def reward(self, action):        
        # at a given step, it is always best to pick the highest chocolate level
        # but, this has long-term issues
        r = 0
        # for i in range(self.slate_size):
        r += (1 - self.sugar_level) * self.documents[action]
        return r
    
    def resample_docs(self):
        self.documents = np.random.rand(self.num_candidates)
    
    def done(self):
        return self.step_count >= self.max_steps

    def step(self, action):
        # Cleaner case for when slate_size=1
        # try: 
        #     action[0]
        # except TypeError:
        #     action = [action]
            
        # Iterate step count
        self.step_count += 1
        
        # Compute reward
        reward = self.reward(action)
        
        # Deal with history
        # BEFORE resampling documents
        self.history_stack[1:] = self.history_stack[:-1] # shift history
        self.history_stack[0] = self.documents[action] # new history
        
        # Update sugar level
        # for i in range(self.slate_size):
        self.sugar_level = self.alpha * self.sugar_level + \
                           (1 - self.alpha) * self.documents[action]
        
        if self.resample_documents:
            self.resample_docs()
        
        return self.observation(), reward, self.done(), {"sugar_level" : self.sugar_level}


class BasicRecommenderWithHistory(BasicRecommender):
    def _set_spaces(self):
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.num_candidates + self.history_len,))
        # if self.slate_size == 1:
        self.action_space = gym.spaces.Discrete(self.num_candidates)
        # else:
            # self.action_space = gym.spaces.MultiDiscrete([self.num_candidates]*self.slate_size) 

    def observation(self):
        return np.concatenate((self.documents, self.history_stack))

