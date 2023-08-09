import numpy as np


class OccupancyMeasure:
    def __init__(self, shape, agent_num):
        self.agent_num = agent_num
        self.lambdas = [np.zeros(shape) for _ in range(agent_num)]
        self.local_lambdas = [np.zeros(shape) for _ in range(agent_num)]

    def update_lambdas(self):
        for aid in range(self.agent_num):
            self.local_lambdas[aid] /= self.local_lambdas[aid].sum()
            self.lambdas[aid] += self.local_lambdas[aid]

    def normalize(self):
        for aid in range(self.agent_num):
            self.lambdas[aid] /= self.lambdas[aid].sum()

    def reset(self):
        for aid in range(self.agent_num):
            self.lambdas[aid] *= 0
            self.local_lambdas[aid] *= 0


class StateOccupancyMeasure(OccupancyMeasure):
    def __init__(self, state_shape, agent_num):
        super().__init__(state_shape, agent_num)

    def count_cur_state(self, state):
        for aid in range(self.agent_num):
            i, j = state[aid]
            self.local_lambdas[aid][i, j] += 1

    def get_prob(self, state):
        prob = [0] * self.agent_num
        for aid in range(self.agent_num):
            i, j = state[aid]
            prob[aid] = self.lambdas[aid][i, j]
        return prob


class StateActionOccupancyMeasure(OccupancyMeasure):
    def __init__(self, state_shape, action_dim, agent_num):
        shape = [action_dim] + state_shape
        super().__init__(shape, agent_num)

    def count_cur_state(self, state, action):
        for aid in range(self.agent_num):
            i, j = state[aid]
            self.local_lambdas[aid][action[aid], i, j] += 1

    def get_prob(self, state, action):
        prob = [0] * self.agent_num
        for aid in range(self.agent_num):
            i, j = state[aid]
            prob[aid] = self.lambdas[aid][action[aid], i, j]
        return prob