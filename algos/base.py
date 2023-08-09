import numpy as np
import torch
import utils

from algos.occupancy_measure import StateOccupancyMeasure
from smac.env import StarCraft2Env


class ExpBuffer:
    def __init__(self, max_len, state_dim, action_dim, agent_num, args):
        self.agent_num = agent_num
        self.use_prior = args.use_prior
        self.use_state_norm = args.use_state_norm
        self.use_local_obs = args.local_obs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        if self.use_local_obs:
            self.state = [torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device) for _ in range(agent_num)]
            self.state_rms = [utils.RunningMeanStd(shape=(state_dim,)) for _ in range(agent_num)]
            self.mask = [torch.ones((max_len, action_dim), dtype=torch.bool, device=self.device) for _ in range(agent_num)]
        else:
            self.state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
            self.state_rms = utils.RunningMeanStd(shape=(state_dim,))
            self.mask = torch.ones((max_len, action_dim), dtype=torch.bool, device=self.device)
        self.action = torch.empty((max_len, agent_num), dtype=torch.float32, device=self.device)
        self.reward = torch.empty((max_len, agent_num + agent_num * self.use_prior), dtype=torch.float32, device=self.device)
        self.done = torch.empty((max_len, 1), dtype=torch.float32, device=self.device)

    def append(self, state, action, reward, done, mask=None):
        if self.now_len >= self.max_len:
            return
        if self.use_local_obs:
            for aid in range(self.agent_num):
                self.state[aid][self.now_len] = torch.as_tensor(state[aid], device=self.device)
                if mask is not None:
                    self.mask[aid][self.now_len] = torch.as_tensor(mask[aid], device=self.device)
        else:
            self.state[self.now_len] = torch.as_tensor(state, device=self.device).flatten()
            if mask is not None:
                self.mask[self.now_len] = torch.as_tensor(mask, device=self.device)
        self.action[self.now_len] = torch.as_tensor(action, device=self.device).flatten()
        self.reward[self.now_len] = torch.as_tensor(reward, device=self.device).flatten()
        self.done[self.now_len] = torch.as_tensor(done, device=self.device)

        self.now_len += 1

    def update_rms(self):
        if self.use_local_obs:
            for aid in range(self.agent_num):
                self.state_rms[aid].update(self.state[aid][:self.now_len])
                print("agent " + str(aid) + ": state mean:", self.state_rms.mean.cpu().numpy(), ",  state variance: ",
                      self.state_rms.var.cpu().numpy())
        else:
            self.state_rms.update(self.state[:self.now_len])
            print("state mean:", self.state_rms.mean.cpu().numpy(), ",  state variance: ", self.state_rms.var.cpu().numpy())

    def normalize_obs(self, state):
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        epsilon = 1e-8
        if self.use_local_obs:
            for aid in range(self.agent_num):
                state[aid] = (state[aid] - self.state_rms[aid].mean) / torch.sqrt(self.state_rms[aid].var + epsilon)
        else:
            state = (state - self.state_rms.mean) / torch.sqrt(self.state_rms.var + epsilon)
        return state

    def sample_batch(self, batch_size):
        indices = torch.randint(self.now_len - 1, size=(batch_size,), device=self.device)
        if self.use_local_obs:
            state = [self.state[aid][indices] for aid in range(self.agent_num)]
        else:
            state = self.state[indices]
        if self.use_state_norm:
            state = self.normalize_obs(state)
        return state, self.reward[indices], self.action[indices], self.done[indices]

    def sample_all(self):
        if self.use_local_obs:
            state = [self.state[aid][:self.now_len] for aid in range(self.agent_num)]
            mask = [self.mask[aid][:self.now_len] for aid in range(self.agent_num)]
        else:
            state = self.state[:self.now_len]
            mask = self.mask[:self.now_len]
        if self.use_state_norm:
            state = self.normalize_obs(state)
        return state, self.reward[:self.now_len], self.action[:self.now_len], self.done[:self.now_len], mask

    def empty_buffer_before_explore(self):
        self.now_len = 0


class AgentBase:
    def __init__(self, env, args, prior=None, share=False):
        self.env = env
        self.agent_num = env.agent_num
        self.use_local_obs = args.local_obs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = 0.0001  # the learning rate for optimizers
        self.gamma = 0.99  #0.99  # the discount factor for future rewards
        self.lambda_gae_adv = 0.98  # could be 0.95 ~ 0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)

        self.clip_grad = args.clip_grad
        self.add_noise = args.add_noise

        self.share_parameter = share

        if self.share_parameter:
            self.acmodel = None
            self.optimizer = None
        else:
            self.acmodels = []
            self.optimizers = []

        if prior is None:
            self.use_prior = False
            self.pweight = 0
        else:
            self.use_prior = True
            self.N = args.N
            self.prior = prior
            self.pweight = args.pweight
            self.pdecay = args.pdecay

        self.use_expert_traj = False

        if self.use_prior:
            self.occupancy_measures = StateOccupancyMeasure(env.grid.shape, env.agent_num)

    def load_status(self, status):
        if self.share_parameter:
            self.acmodel.load_state_dict(status["model_state"])
            self.optimizer.load_state_dict(status["optimizer_state"])
        else:
            for aid in range(self.agent_num):
                self.acmodels[aid].load_state_dict(status["model_state"][aid])
                self.optimizers[aid].load_state_dict(status["optimizer_state"][aid])

    def select_action(self, state, mask):
        pass

    def get_prior_prob(self, state, action):
        prob = [0] * self.agent_num
        for aid in range(self.agent_num):
            prob[aid] = self.prior[aid][action[aid], state[aid][0], state[aid][1]]
        return prob

    def compute_shadow_r(self, state, action):
        shadow_r = [0] * self.agent_num
        cur_probs = self.occupancy_measures.get_prob(state)
        prior_probs = self.get_prior_prob(state, action)
        for aid in range(self.agent_num):
            cur_prob = cur_probs[aid] + 1e-12
            prior_prob = prior_probs[aid] + 1e-12
            if cur_prob != prior_prob:
                shadow_r[aid] = - (np.log(2 * cur_prob) - np.log(cur_prob + prior_prob))
        return shadow_r

    def compute_lambda(self):
        episode = 0
        while episode < self.N:
            state = self.env.reset()
            self.occupancy_measures.count_cur_state(state["vec"])
            done = False
            while not done:
                action = self.select_action(state["vec"], state.get("mask"))
                state, reward, done, _ = self.env.step(action)
                self.occupancy_measures.count_cur_state(state["vec"])
            self.occupancy_measures.update_lambdas()
            episode += 1
        self.occupancy_measures.normalize()

