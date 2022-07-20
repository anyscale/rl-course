import numpy as np

def my_render_frozen_lake(self):
    
    original = True # original frozen lake letter
    
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
                print("H" if original else "O", end="")
            else:
                print("F" if original else ".", end="")
        print()

def fix_frozen_lake_render(env):
    env.render = type(env.render)(my_render_frozen_lake, env)
    
def my_render_cartpole(self):
    POLE_LENGTH = 12
    SCREEN_WIDTH = 50
    x, x_dot, theta, theta_dot = self.state
    
    top_x_displacement = np.sin(theta) * POLE_LENGTH
    top_x_displacement = np.round(top_x_displacement)
    top_x_loc = SCREEN_WIDTH//2 + top_x_displacement
    
    top_y_loc = np.cos(theta) * POLE_LENGTH
    top_y_loc = np.round(top_y_loc)
    
    # bottom always at (0,0)
    screen = np.zeros((POLE_LENGTH, SCREEN_WIDTH), dtype=bool)
    
    

def fix_cartpole_render(env):
    env.render = type(env.render)(my_render_cartpole, env)

    
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