import os
import time
import numpy as np
from envs.gridworld import GridWorldEnv
import utils
from envfiles.funcs.utils import *
import torch
from torch.distributions.categorical import Categorical


def select_action(state, prior):
    actions = prior[:, state[0], state[1]].flatten()
    if actions.sum() > 0:
        actions /= actions.sum()
    action_dist = Categorical(torch.tensor(actions))
    return action_dist.sample()


def save_to_file(data, file_path):
    try:
        with open(file_path, "ab") as handle:
            np.savetxt(handle, (data,), fmt="%s")
    except FileNotFoundError:
        with open(file_path, "wb") as handle:
            np.savetxt(handle, (data,), fmt="%s")


if __name__ == '__main__':

    aid = 4

    env_name = "centerSquare6x6_1a_" + str(aid)
    env = GridWorldEnv(env_name, True)

    dir_name = generate_dir(env_name.split("_")[0])
    prior = np.load("envfiles/" + dir_name + "_prior" + str(aid) + ".npy")

    EPISODES = 200
    STEPS = 100
    RENDER = False
    SAVE = True

    for episode in range(EPISODES):
        done = False
        state = env.reset()
        lp = []
        r = []

        for step in range(STEPS):
            if RENDER:
                img = env.render()
            action = select_action(state["vec"].flatten(), prior)

            if SAVE:
                save_to_file(state["vec"].flatten(), "trajs/centerSquare6x6_states" + str(aid) + ".csv")
                save_to_file([action.numpy()], "trajs/centerSquare6x6_actions" + str(aid) + ".csv")

            state, r_, done, i_ = env.step(action)

            if done or (env.window is not None and env.window.closed):
                state = env.reset()
                break

        if env.window is not None and env.window.closed:
            break
