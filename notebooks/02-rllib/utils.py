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
    x, x_dot, theta, theta_dot = self.state
    print(x)
    print(theta)
    
    POLE_LENGTH = 12
    SCREEN_WIDTH = 60

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