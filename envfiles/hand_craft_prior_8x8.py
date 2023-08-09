import numpy as np
import json
from enum import IntEnum
import matplotlib.pyplot as plt
from envs.gridworld import GridWorldEnv
from envfiles.funcs.utils import *
from envfiles.show_crafted_policy import visualize_policy

np.set_printoptions(precision=3)


class Actions(IntEnum):
    left = 0
    right = 1
    up = 2
    down = 3
    stay = 4


env_name = "centerSquare8x8"
dir_name = generate_dir(env_name)
env = GridWorldEnv(env_name + "_4a")

state_lambdas = np.zeros(env.grid.shape)

num_visit = (env.lava == 0).sum()
state_lambdas[env.lava == 0] = 1 / num_visit

actions = [[Actions.right, Actions.up],
           [Actions.left, Actions.up],
           [Actions.left, Actions.down],
           [Actions.right, Actions.down]]

for aid in range(4):
    sa_lambdas = np.zeros((5,) + env.grid.shape)
    # for agent:
    agent_i, agent_j = env.agents[aid]
    sa_lambdas[actions[aid][0], agent_i, agent_j] = state_lambdas[agent_i, agent_j] * 0.5
    sa_lambdas[actions[aid][1], agent_i, agent_j] = state_lambdas[agent_i, agent_j] * 0.5
    for j in range(1, env.width-1):
        sa_lambdas[actions[aid][0], agent_i, j] = state_lambdas[agent_i, j]
    for i in range(1, env.height-1):
        sa_lambdas[actions[aid][1], i, agent_j] = state_lambdas[i, agent_j]

    # for goal:
    goal_i, goal_j = env.goals[aid]
    for j in range(env.width):
        sa_lambdas[actions[aid][0], goal_i, j] = state_lambdas[goal_i, j]
    for i in range(env.height):
        sa_lambdas[actions[aid][1], i, goal_j] = state_lambdas[i, goal_j]

    sa_lambdas[0:4, goal_i, goal_j] = state_lambdas[goal_i, goal_j] * 0.25

    print(state_lambdas.sum())
    print(sa_lambdas.sum())
    print(sa_lambdas.sum(0))

    np.save(dir_name + "_prior" + str(aid) + ".npy", sa_lambdas)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks(np.arange(0, 10+1, 1))
    ax.set_yticks(np.arange(0, 10+1, 1))
    ax.set_aspect('equal')
    visualize_policy(env.agents[aid], env.goals[aid], sa_lambdas)

    plt.savefig(dir_name + "_policy" + str(aid) + '.jpg', bbox_inches='tight')
    plt.close(fig)
