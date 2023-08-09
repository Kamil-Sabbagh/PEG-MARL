import numpy as np
import matplotlib.pyplot as plt
from envs.gridworld import GridWorldEnv
from envs.appledoor import AppleDoorEnv
from envfiles.funcs.utils import *


def visualize_policy(agent_pose, goal_pose, prior):
    d, h, w = prior.shape

    plt.grid(True, axis='both', color='black', alpha=0.5, linestyle='--')
    plt.xlim(0, w)
    plt.ylim(0, h)

    # agent
    self_y, self_x = agent_pose
    plt.fill([self_x, self_x, self_x + 1, self_x + 1], [self_y, self_y + 1, self_y + 1, self_y], facecolor='red')

    # goal
    goal_y, goal_x = goal_pose
    plt.fill([goal_x, goal_x, goal_x + 1, goal_x + 1], [goal_y, goal_y + 1, goal_y + 1, goal_y], facecolor='green')


    for i in range(h):
        for j in range(w):
            actions = prior[:, i, j]
            if prior[:, i, j].sum() > 0:
                actions /= prior[:, i, j].sum()

            self_y, self_x = i, j
            if actions[0] > 0:
                plt.arrow(self_x + 0.5, self_y + 0.5, -0.4 * actions[0], 0, facecolor='k', ec='k', head_width=0.1,
                          head_length=0.1)
            if actions[1] > 0:
                plt.arrow(self_x + 0.5, self_y + 0.5, 0.4 * actions[1], 0, facecolor='k', ec='k', head_width=0.1,
                          head_length=0.1)
            if actions[2] > 0:
                plt.arrow(self_x + 0.5, self_y + 0.5, 0, 0.4 * actions[2], facecolor='k', ec='k', head_width=0.1,
                          head_length=0.1)
            if actions[3] > 0:
                plt.arrow(self_x + 0.5, self_y + 0.5, 0, -0.4 * actions[3], facecolor='k', ec='k', head_width=0.1,
                          head_length=0.1)


if __name__ == '__main__':
    # env_name = "centerSquare8x8"
    # dir_name = generate_dir(env_name)

    # env_name = "centerSquare6x6"
    # dir_name = "../priors/centerSquare6x6_suboptimal"
    # env = GridWorldEnv(env_name + "_4a")

    env_name = "appleDoor_a"
    dir_name = "../priors/appleDoor_a"
    env = AppleDoorEnv(env_name)
    for aid in range(len(env.agents)):
        prior = np.load(dir_name + "_suboptimal_prior" + str(aid) + ".npy")
        # prior = np.load(env_name.split("_")[0] +"/" + env_name.split("_")[0] + "_prior.npy")

        state_prior = prior.sum(0)

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 2, 1)
        ax.set_xticks(np.arange(0, 10+1, 1))
        ax.set_yticks(np.arange(0, 10+1, 1))
        ax.set_aspect('equal')

        visualize_policy(env.agents[aid], env.goals[aid], prior)

        ax = fig.add_subplot(1, 2, 2)

        plt.imshow(np.flip(state_prior, axis=0))

        # plt.show()
        plt.savefig(dir_name + "_suboptimal_prior" + str(aid) + '.jpg', bbox_inches='tight')
        plt.close(fig)
