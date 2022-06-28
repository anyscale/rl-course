class SlateRecommender(gym.Env):
    def __init__(self, env_config=None):
        
        if env_config is None:
            env_config = dict()
            
        self.num_candidates = env_config.get("num_candidates", 10)
        self.slate_size = env_config.get("slate_size", 1)
        self.resample_documents = env_config.get("resample_documents", True)
        self.max_steps = env_config.get("max_steps", 100)
        self.sugar_momentum = env_config.get("sugar_momentum", 0.8) # (0,1)
        
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.num_candidates,))
        
        # if self.slate_size == 1:
        self.action_space = gym.spaces.Discrete(self.num_candidates)
        # else:
            # self.action_space = gym.spaces.MultiDiscrete([self.num_candidates]*self.slate_size) 
        
        # Create the documents
        if "seed" in env_config:
            self.seed(env_config["seed"])
        self.resample_docs()

    def seed(self, seed):
        np.random.seed(seed)
        
    def reset(self):
        self.sugar_level = 0.0 # starts with no satiety 
        self.step_count = 0
        
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
        
        # Update sugar level
        # for i in range(self.slate_size):
        self.sugar_level = self.sugar_momentum * self.sugar_level + (1 - self.sugar_momentum) * self.documents[action]
        
        if self.resample_documents:
            self.resample_docs()
        
        return self.observation(), reward, self.done(), {"sugar_level" : self.sugar_level}