import numpy as np
import torch
from ray.rllib.models.preprocessors import get_preprocessor 
from ray.rllib.algorithms.ppo import PPOConfig


def query_policy(trainer, env, obs, actions=None):
    policy = trainer.get_policy()
    model = policy.model    
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    model_output = model({"obs": torch.from_numpy(prep.transform(obs)[None])})[0]
    dist = policy.dist_class(model_output, model) 
    if actions is None:
        actions = [0, 1, 2, 3]
    probs = np.exp(dist.logp(torch.from_numpy(np.array(actions))).detach().numpy())
    return probs

# lake_default_config = {
#     "framework"             : "torch",
#     "create_env_on_driver"  : True,
#     "seed"                  : 0,
#     "horizon"               : 100,
#     "model"                 : {"fcnet_hiddens" : [32, 32]}
# }

lake_default_config = (
    PPOConfig()
    .framework("torch")
    .rollouts(create_env_on_local_worker=True, horizon=100)
    .debugging(seed=0, log_level="ERROR")
    .training(model={"fcnet_hiddens": [32, 32]})
)
