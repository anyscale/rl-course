import numpy as np
import torch
from ray.rllib.models.preprocessors import get_preprocessor 

def query_policy(trainer, env, obs, actions=None):
    policy = trainer.get_policy()
    model = policy.model    
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    model_output = model({"obs": torch.from_numpy(prep.transform(obs)[None])})[0]
    dist = policy.dist_class(model_output, model) 
    if actions is None:
        actions = [0,1,2,3]
    probs = np.exp(dist.logp(torch.from_numpy(np.array(actions))).detach().numpy())
    return probs

# TODO: don't repeat this code from module 1 utils
def my_render(self):
    
    if self.lastaction is not None:
        print(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})")
    else:
        print("")
    
    desc = self.desc.tolist()

    row, col = self.s // self.ncol, self.s % self.ncol
    desc = [[c.decode("utf-8") for c in line] for line in desc]

    for i, line in enumerate(desc):
        for j, c in enumerate(line):
            if (i,j) == (row, col):
                print("P", end="")
            elif desc[i][j] == "G":
                print("G", end="")
            elif desc[i][j] == "H":
                print("O", end="")
            else:
                print(".", end="")
        print()

def fix_frozen_lake_render(env):
    env.render = type(env.render)(my_render, env)
    