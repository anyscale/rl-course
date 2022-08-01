import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from IPython import display
import torch
from ray.rllib.models.preprocessors import get_preprocessor 
from ray.rllib.algorithms.ppo import PPO, PPOConfig

import pandas as pd

# slippery_algo_config = {
#     "framework"             : "torch",
#     "create_env_on_driver"  : True,
#     "seed"                  : 0,
#     "env_config"            : {"is_slippery" : True},
#     "evaluation_config"     : {"explore" : False}
# }

slippery_algo_config = (
    PPOConfig()\
    .framework("torch")\
    .rollouts(create_env_on_local_worker=True)\
    .debugging(seed=0, log_level="ERROR")\
    .training(model={"fcnet_hiddens" : [64,64]})
    .environment(env_config={"is_slippery" : True})\
    .evaluation(evaluation_config = {"explore" : False})
)


def my_render_frozen_lake(self):
    
    original = False # original frozen lake letter
    
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
                # print("🧑", end="")
            elif desc[i][j] == "G":
                print("G", end="")
                # print("⛳️", end="")
            elif desc[i][j] == "H":
                print("H" if original else "O", end="")
                # print("🕳", end="")
            else:
                print("F" if original else ".", end="")
                # print("🧊", end="")
        print()

def fix_frozen_lake_render(env):
    env.render = type(env.render)(my_render_frozen_lake, env)
    
def my_render_cartpole_text(self):
    x, x_dot, theta, theta_dot = self.state
    theta *= 2 # FOR RENDERING MORE DRAMATiCALLY
    
    POLE_LENGTH = 12
    SCREEN_WIDTH = 30

    screen = np.full((POLE_LENGTH+1, SCREEN_WIDTH+1), " ")

    yy = np.arange(POLE_LENGTH+1)
    xx = np.arange(SCREEN_WIDTH+1) - SCREEN_WIDTH//2
    angles = np.arctan2(xx-x, yy[:,None])
    min_inds = np.argmin(np.abs(angles-theta), axis=1)

    screen[np.arange(POLE_LENGTH+1, dtype=int), min_inds] = "."
    screen[0] = " "
    screen[0, int(np.round(x + SCREEN_WIDTH//2))] = "O"#"⬛️"

    screen = screen[-1::-1,:]

    for line in screen:
        s = "".join(line)
        flag = False
        print(s)
        
def my_render_cartpole_matplotlib(self):
    x, x_dot, theta, theta_dot = self.state
    
    pole_length = 3
    base_height = 0.075 * pole_length
    base_width = 0.75

    if not hasattr(self, "render_ax"):
        fig = plt.figure()
        self.render_fig = fig
    self.render_ax = plt.gca()
    ax = self.render_ax
    ax.clear()
    ax.set_xticks([-4.8, -2.4, 0, 2.4, 4.8]);
    ax.set_yticks([]);
    ax.set_xlim([-5,5]);
    ax.set_ylim([-base_height-0.02, pole_length + 0.2]);
    ax.set_aspect('equal')
    ax.plot([x,x+np.sin(theta)*pole_length], [0,np.cos(theta)*pole_length], linewidth=4, color="goldenrod");
    rect = matplotlib.patches.Rectangle((x-base_width/2, -base_height), 
                base_width, base_height, linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(rect);
    ax.set_title(f"angle = {theta*180/np.pi:.2f} degrees");
    # plt.show();
    display.clear_output(wait=True);
    display.display(self.render_fig);

def fix_cartpole_render(env):
    env.render = type(env.render)(my_render_cartpole_matplotlib, env)

    
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



def plot_action_probs(action_probs):
    df = pd.DataFrame(action_probs, index=["left", "down", "right", "up"]).T
    plt.rcParams["font.size"] = 14
    plt.figure(figsize=(3,6))
    plt.imshow(df.values);
    plt.xticks(np.arange(4), labels=["left", "down", "up", "right"]);
    plt.yticks(np.arange(16), labels=np.arange(16));
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor");
    plt.ylabel("observations");
    plt.xlabel("actions");
    cbar = plt.colorbar();
    cbar.ax.set_ylabel('action probability', rotation=270)
    cbar.set_ticks([0, 0.25, 0.75, 1])
    plt.tight_layout();