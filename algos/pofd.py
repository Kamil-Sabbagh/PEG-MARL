import torch
import torch.nn as nn
import numpy as np
from algos.ppo import PPO
from algos.model import Discriminator


class POfD(PPO):
    def __init__(self, env, args, expert_traj, target_steps=2048):
        super().__init__(env, args, target_steps, prior=None)

        self.exp_state_dim = expert_traj["states"][0].shape[1]
        self.action_num = env.action_space[0].n

        self.use_expert_traj = True
        self.pweight = args.pweight
        self.pdecay = args.pdecay

        self.expert_state_actions = []
        self.load_expert_trajectory(expert_traj)

        self.bce_loss = nn.BCELoss()

        self.num_d_epochs = 2
        self.discriminators = []
        self.d_optimizers = []
        for aid in range(env.agent_num):
            self.discriminators.append(Discriminator(self.exp_state_dim, self.action_num))
            self.discriminators[aid].to(self.device)
            self.d_optimizers.append(torch.optim.Adam(self.discriminators[aid].parameters(), self.lr))

    def load_status(self, status):
        for aid in range(self.agent_num):
            self.acmodels[aid].load_state_dict(status["model_state"][aid])
            self.optimizers[aid].load_state_dict(status["optimizer_state"][aid])
            self.discriminators[aid].load_state_dict(status["discriminator_state"][aid])
            self.d_optimizers[aid].load_state_dict(status["d_optimizer_state"][aid])

    def load_expert_trajectory(self, expert):
        for aid in range(self.agent_num):
            expert_states = torch.tensor(expert["states"][aid], dtype=torch.float32, device=self.device)
            expert_actions = torch.tensor(expert["actions"][aid], dtype=torch.int64, device=self.device)
            expert_actions = torch.eye(self.action_num, device=self.device)[expert_actions]
            self.expert_state_actions.append(torch.cat([expert_states, expert_actions], dim=1))

    def normalize_expert(self, rms):
        epsilon = 1e-8
        expert_state_actions = []
        for aid in range(self.agent_num):
            if self.use_local_obs:
                cur_rms = rms[aid]
            else:
                cur_rms = rms
            sid = self.exp_state_dim * aid
            cur_mean = cur_rms.mean[sid:sid+self.exp_state_dim]
            cur_var = cur_rms.var[sid:sid+self.exp_state_dim]
            cur_sa = self.expert_state_actions[aid].clone().detach()
            cur_sa[:, :self.exp_state_dim] = (cur_sa[:, :self.exp_state_dim] - cur_mean) / torch.sqrt(cur_var + epsilon)
            expert_state_actions.append(cur_sa)
        return expert_state_actions

    def concatenate_state_actions(self, states, actions):
        state_actions = []
        for aid in range(self.agent_num):
            actions = torch.eye(self.action_num, device=self.device)[actions[:, aid].to(torch.int64)]
            sid = self.exp_state_dim * aid
            if self.use_local_obs:  # mpe simple spread environment
                cur_state = states[aid][:, :self.exp_state_dim]
            else:   # gridworld environment
                cur_state = states[:, sid:sid+self.exp_state_dim]
            state_actions.append(torch.cat([cur_state, actions], dim=1))
        return state_actions

    def update_parameters(self, buffer, tb_writer=None):
        buf_len = buffer.now_len
        buf_state, buf_reward, buf_action, buf_done, buf_mask = buffer.sample_all()

        buf_state_actions = self.concatenate_state_actions(buf_state, buf_action)
        if self.use_state_norm:
            cur_expert_sas = self.normalize_expert(buffer.state_rms)
        else:
            cur_expert_sas = self.expert_state_actions
        for ep in range(self.num_d_epochs):
            for aid in range(self.agent_num):
                expert_prob = self.discriminators[aid](cur_expert_sas[aid])
                agent_prob = self.discriminators[aid](buf_state_actions[aid])
                term1 = self.bce_loss(agent_prob, torch.ones((buf_state_actions[aid].shape[0], 1), device=self.device))
                term2 = self.bce_loss(expert_prob, torch.zeros((cur_expert_sas[aid].shape[0], 1), device=self.device))
                loss = term1 + term2
                if tb_writer:
                    tb_writer.tb_writer.add_scalar("ep_d_loss" + str(aid), loss, tb_writer.ep_num)
                    tb_writer.tb_writer.add_scalar("ep_prob_agent" + str(aid), agent_prob.mean(), tb_writer.ep_num)
                    tb_writer.tb_writer.add_scalar("ep_prob_expert" + str(aid), expert_prob.mean(), tb_writer.ep_num)
                self.d_optimizers[aid].zero_grad()
                loss.backward()
                self.d_optimizers[aid].step()

        with torch.no_grad():
            d_rewards = -torch.log(
                torch.hstack([self.discriminators[aid](buf_state_actions[aid]) for aid in range(self.agent_num)]))
            if tb_writer:
                d_rewards_mean = d_rewards.mean(axis=0)
                for aid in range(self.agent_num):
                    tb_writer.tb_writer.add_scalar("ep_d_rewards" + str(aid), d_rewards_mean[aid], tb_writer.ep_num)
            buf_reward = torch.hstack([buf_reward, d_rewards])
            buf_value, buf_logprob = self.batch_collect_value(buf_state, buf_action, buf_mask)
            buf_r_sum, buf_advantage = self.compute_return_adv(buf_len, buf_reward, buf_done, buf_value)
            del buf_reward, buf_done

        self.update_policy_critic(buf_state, buf_action, buf_mask, buf_value, buf_logprob, buf_r_sum, buf_advantage, tb_writer)
