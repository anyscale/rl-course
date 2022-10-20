import numpy as np
import gym

class BasicRecommender(gym.Env):
    def __init__(self, env_config=None):
        
        if env_config is None:
            env_config = dict()
            
        self.num_candidates = env_config.get("num_candidates", 10)
        self.slate_size = env_config.get("slate_size", 1)
        self.resample_documents = env_config.get("resample_documents", True)
        self.max_steps = env_config.get("max_steps", 100)
        self.alpha = env_config.get("alpha", 0.9) # (0,1)
        self.history_len = env_config.get("history_len", 2) # number of past frames (not including present)
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

