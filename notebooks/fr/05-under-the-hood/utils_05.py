from ray.rllib.models.preprocessors import get_preprocessor 
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 16

def get_q_state(algo, env, obs):
    model = algo.get_policy().model
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    model_out = model({"obs": torch.from_numpy(prep.transform(obs)[None])})[0]
    return float(model.get_state_value(model_out))


def q_state_plot_frozenlake(q_s, env):
    plt.imshow(np.reshape(q_s, (4,4)));
    plt.colorbar();
    plt.xticks(());
    plt.yticks(());
    desc = [[c.decode("utf-8") for c in line] for line in env.desc.tolist()]
    mapper = {"F" : ".", "H" : "O", "S" : "S", "G" : "G"}
    for i in range(4):
        for j in range(4):
            plt.text(j-0.1,i+0.1, mapper[desc[i][j]], fontsize=25, color="white")
            
            
def get_q_state_action(algo, env, obs):
    model = algo.get_policy().model
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    model_out = model({"obs": torch.from_numpy(prep.transform(obs)[None])})[0]
    return model.get_q_value_distributions(model_out)[0].detach().numpy()[0]


# https://stackoverflow.com/questions/44666679/something-like-plt-matshow-but-with-triangles
def quatromatrix(left, bottom, right, top, ax=None, triplotkw={},tripcolorkw={}):
    if not ax: ax=plt.gca()
    n = left.shape[0]; m=left.shape[1]

    a = np.array([[0,0],[0,1],[.5,.5],[1,0],[1,1]])
    tr = np.array([[0,1,2], [0,2,3],[2,3,4],[1,2,4]])

    A = np.zeros((n*m*5,2))
    Tr = np.zeros((n*m*4,3))

    for i in range(n):
        for j in range(m):
            k = i*m+j
            A[k*5:(k+1)*5,:] = np.c_[a[:,0]+j, a[:,1]+i]
            Tr[k*4:(k+1)*4,:] = tr + k*5

    C = np.c_[ left.flatten(), bottom.flatten(), 
              right.flatten(), top.flatten()   ].flatten()

    triplot = ax.triplot(A[:,0], A[:,1], Tr, **triplotkw)
    tripcolor = ax.tripcolor(A[:,0], A[:,1], Tr, facecolors=C, **tripcolorkw)
    # return tripcolor
    return ax


def q_state_action_plot_frozenlake(q_sa, env):
    ax = quatromatrix(*q_sa.reshape((4,4,4)).transpose((2,0,1))[[0,3,2,1]])
    # above line:
    # first, reshape to 4x4 of arena, with last 4 dimension for the action space (left, down, up, right)
    # then, change the action space dimension to be first with the transpose, to use with *
    # then, we actually need to switch the up and down because of the invert_yaxis,
    # because from the perspective of the map, up is down
    ax.set_xticks(());
    ax.set_yticks(());
    ax.invert_yaxis();
    desc = [[c.decode("utf-8") for c in line] for line in env.desc.tolist()]
    mapper = {"F" : ".", "H" : "O", "S" : "S", "G" : "G"}
    for i in range(4):
        for j in range(4):
            plt.text(j+0.4,i+0.7, mapper[desc[i][j]], fontsize=25, color="white")
    # fig = plt.gcf()

    norm = mpl.colors.Normalize(vmin=q_sa.min(), vmax=q_sa.max())
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax);