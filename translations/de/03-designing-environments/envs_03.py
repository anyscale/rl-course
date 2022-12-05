import gym
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


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
    