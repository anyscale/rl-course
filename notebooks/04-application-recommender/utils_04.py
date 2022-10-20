import numpy as np
import torch
from ray.rllib.models.preprocessors import get_preprocessor 
import json


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


def load_offline_data(filename):
    rollouts = []
    # TODO this variable does not exist
    with open(offline_recsys_file, "r") as f:
        for line in f:
            data = json.loads(line)
            rollouts.append(data)
    return rollouts