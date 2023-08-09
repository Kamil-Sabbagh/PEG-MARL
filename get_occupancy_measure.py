import numpy as np
from envs.gridworld import GridWorldEnv


if __name__ == '__main__':
    for aid in range(2):
        # env_name = "centerSquare6x6_1a_" + str(aid)
        env_name = "appleDoor_a_" + str(aid + 1)
        env = GridWorldEnv(env_name)

        # states = np.genfromtxt("trajs/centerSquare6x6_suboptimal_states{0}.csv".format(aid), dtype=np.int32)
        # actions = np.genfromtxt("trajs/centerSquare6x6_suboptimal_actions{0}.csv".format(aid), dtype=np.int32)

        states = np.genfromtxt("trajs/appleDoor_a_suboptimal_states{0}.csv".format(aid), dtype=np.int32)
        actions = np.genfromtxt("trajs/appleDoor_a_suboptimal_actions{0}.csv".format(aid), dtype=np.int32)

        lambdas = np.zeros([5, env.height, env.width])

        for i in range(len(states)):
            x, y = states[i]
            a = actions[i]
            lambdas[a, x, y] += 1

        lambdas /= lambdas.sum()

        print(lambdas.sum())

        # np.save("priors/centerSquare6x6_suboptimal_prior" + str(aid) + ".npy", lambdas)
        np.save("priors/appleDoor_a_suboptimal_prior" + str(aid) + ".npy", lambdas)
