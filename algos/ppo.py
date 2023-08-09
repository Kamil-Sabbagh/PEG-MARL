import torch
import numpy as np
import time
import utils
from algos.base import AgentBase
from algos.model import ACModel, SharedForwardModel  # Import the SharedForwardModel
from smac.env import StarCraft2Env
from torch import optim  # Import the optim module for the optimizer
import torch.nn.functional as F  # Import the functional module for the loss function



class PPO(AgentBase):
    def __init__(self, env, args, target_steps=2048, prior=None):
        super().__init__(env, args, prior)
        self.use_gae = args.use_gae
        self.use_state_norm = args.use_state_norm
        self.use_value_norm = args.use_value_norm
        self.share_reward = False
        self.param_share = True

        self.is_mpe = "mpe" in args.env

        self.target_steps = target_steps
        self.ppo_epoch = args.ppo_epoch  # how many times to reuse the memory
        self.num_mini_batch = args.num_mini_batch  # how many frames for each update
        self.batch_size = target_steps / args.num_mini_batch

        self.clip_eps = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.01  # could be 0.02
        self.value_loss_coef = 0.5
        self.max_grad_norm = 1

        if self.use_value_norm:
            self.value_normalizer = utils.ValueNorm(self.agent_num, device=self.device)

        # Define the dimensions of the state and action spaces
        state_dim = env.observation_space[0].shape[0]
        action_dim = env.action_space[0].n

        # Define the dimension of the hidden layer of the shared forward model
        hidden_dim = 128  # This should be adjusted based on your specific needs

        # Initialize the shared forward model
        self.shared_forward_model = SharedForwardModel(state_dim, action_dim, hidden_dim).to(self.device)


        for aid in range(env.agent_num):
            self.acmodels.append(ACModel(env.observation_space[aid], env.action_space[aid]))
            self.acmodels[aid].to(self.device)
            # Add the parameters of the shared forward model to the optimizer
            self.optimizers.append(optim.Adam(list(self.acmodels[aid].parameters()) + list(self.shared_forward_model.parameters()), self.lr))


    def select_action(self, state, mask):
        actions = [0] * self.agent_num
        for aid in range(self.agent_num):
            if self.use_local_obs:
                cur_state = state[aid]
                if mask is not None:
                    mask = mask[aid]
            else:
                cur_state = state.flatten()
            dist, value = self.acmodels[aid](cur_state, mask)
            action = dist.sample()
            actions[aid] = action
        return actions

    def batch_collect_value(self, buf_state, buf_action, buf_mask):
        buf_value = torch.zeros(buf_action.shape, device=self.device)
        buf_logprob = torch.zeros(buf_action.shape, device=self.device)
        for aid in range(self.agent_num):
            if self.use_local_obs:
                cur_buf_state = buf_state[aid]
                if buf_mask is not None:
                    cur_buf_mask = buf_mask[aid]
            else:
                cur_buf_state = buf_state
                cur_buf_mask = buf_mask
            dist, value = self.acmodels[aid](cur_buf_state, cur_buf_mask)
            logprob = dist.log_prob(buf_action[:, aid])
            buf_value[:, aid] = value
            buf_logprob[:, aid] = logprob
        return buf_value, buf_logprob

    def collect_experiences(self, buffer, tb_writer=None):
        if self.use_prior:
            self.compute_lambda()

        buffer.empty_buffer_before_explore()
        steps = 0
        saveMore = False and self.is_mpe
        ep_returns = np.zeros(self.agent_num * (1 + (self.use_prior or saveMore)))
        stime = time.time()
        while steps < self.target_steps:
            state = self.env.reset()
            done = False
            ep_steps = 0
            ep_returns *= 0
            while not done:
                # self.env.render()
                action = self.select_action(state["vec"], state.get("mask"))
                next_state, reward, done, info = self.env.step(action)
                if self.use_prior:
                    shadow_reward = self.compute_shadow_r(state["vec"], action)
                    reward = reward + shadow_reward
                buffer.append(state["vec"], action, reward, done, state.get("mask"))

                if saveMore and info.get("n"):
                    reward = reward + info["n"]
                ep_returns += reward
                state = next_state
                steps += 1
                ep_steps += 1
            if tb_writer:
                tb_writer.add_info(ep_steps, ep_returns, self.pweight)
        etime = time.time()
        fps = steps / (etime - stime)
        print("FPS: ", fps)
        if self.use_state_norm:
            buffer.update_rms()
        return steps

def update_parameters(self, buffer, tb_writer=None):
    buf_len = buffer.now_len
    with torch.no_grad():
        buf_state, buf_reward, buf_action, buf_done, buf_mask = buffer.sample_all()
        buf_value, buf_logprob = self.batch_collect_value(buf_state, buf_action, buf_mask)
        buf_r_sum, buf_advantage = self.compute_return_adv(buf_len, buf_reward, buf_done, buf_value)
        del buf_reward, buf_done

    # Compute the predicted next global state
    next_global_state_pred = self.shared_forward_model(buf_state, buf_action)

    # Compute the intrinsic reward as the mean squared error between the predicted next global state and the actual next global state
    intrinsic_reward = (next_global_state_pred - buf_state).pow(2).mean()

    # Add the intrinsic reward to the extrinsic reward to get the total reward
    buf_r_sum += intrinsic_reward

    # Compute the loss
    forward_loss = F.mse_loss(next_global_state_pred, buf_state)

    # Backpropagate the loss
    self.optimizer.zero_grad()
    forward_loss.backward()
    self.optimizer.step()

    self.update_policy_critic(buf_state, buf_action, buf_mask, buf_value, buf_logprob, buf_r_sum, buf_advantage, tb_writer)

    def update_policy_critic(self, buf_state, buf_action, buf_mask, buf_value, buf_logprob, buf_r_sum, buf_advantage, tb_writer):
        buf_len = buf_value.shape[0]
        for i in range(self.ppo_epoch):
            length = int(buf_len // self.num_mini_batch * self.num_mini_batch)
            indices = torch.randperm(length, requires_grad=False, device=self.device).reshape(
                [self.num_mini_batch, int(length / self.num_mini_batch)])
            for ind in indices:
                if self.use_local_obs:
                    sb_state = [buf_state[aid][ind] for aid in range(self.agent_num)]
                    sb_mask = [buf_mask[aid][ind] for aid in range(self.agent_num)]
                else:
                    sb_state = buf_state[ind]
                    sb_mask = buf_mask[ind]
                sb_action = buf_action[ind]
                sb_value = buf_value[ind]
                sb_r_sum = buf_r_sum[ind]
                sb_logprob = buf_logprob[ind]
                sb_advantage = buf_advantage[ind]

                for aid in range(self.agent_num):
                    if self.use_local_obs:
                        cur_sb_state = sb_state[aid]
                        cur_sb_mask = sb_mask[aid]
                    else:
                        cur_sb_state = sb_state
                        cur_sb_mask = sb_mask
                    dist, value = self.acmodels[aid](cur_sb_state, cur_sb_mask)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb_action[:, aid]) - sb_logprob[:, aid])
                    surr1 = sb_advantage[:, aid] * ratio
                    surr2 = sb_advantage[:, aid] * torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = value + torch.clamp(value - sb_value[:, aid], -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb_r_sum[:, aid]).pow(2)
                    surr2 = (value_clipped - sb_r_sum[:, aid]).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    # Compute the predicted next global state
                    next_global_state_pred = self.shared_forward_model(cur_sb_state, sb_action[:, aid])

                    # Compute the intrinsic reward as the mean squared error between the predicted next global state and the actual next global state
                    intrinsic_reward = (next_global_state_pred - cur_sb_state).pow(2).mean()

                    # Add the intrinsic reward to the extrinsic reward to get the total reward
                    sb_r_sum[:, aid] += intrinsic_reward

                    loss = policy_loss - self.lambda_entropy * entropy + self.value_loss_coef * value_loss
                    self.optimizers[aid].zero_grad()
                    loss.backward()
                    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodels[aid].parameters()) ** 0.5
                    if self.clip_grad:
                        torch.nn.utils.clip_grad_norm_(self.acmodels[aid].parameters(), self.max_grad_norm)
                    if self.add_noise and grad_norm < 1:
                        for params in self.acmodels[aid].parameters():
                            params.grad += torch.randn(params.grad.shape, device=self.device)
                    self.optimizers[aid].step()
                    if tb_writer:
                        tb_writer.add_grad_info(aid, policy_loss.item(), value_loss.item(), grad_norm)

        if self.param_share and self.agent_num > 1:
            state_dict_all = [self.acmodels[aid].critic.state_dict() for aid in range(self.agent_num)]
            avg_sd = state_dict_all[0].copy()
            for key in state_dict_all[0]:
                avg_sd[key] = torch.mean(torch.stack([state_dict_all[aid][key] for aid in range(self.agent_num)]),
                                        dim=0)
            for aid in range(self.agent_num):
                self.acmodels[aid].critic.load_state_dict(avg_sd)

    def compute_return_adv(self, buf_len, buf_reward, buf_done, buf_value) -> (torch.Tensor, torch.Tensor):
        if self.use_value_norm:
            buf_value = self.value_normalizer.denormalize(buf_value)
        if self.use_prior or self.use_expert_traj:
            buf_reward = self.reward_shaping(buf_reward)
        if self.use_gae:
            buf_r_sum, buf_advantage = self.compute_reward_gae(buf_len, buf_reward, buf_done, buf_value)
        else:
            buf_r_sum, buf_advantage = self.compute_reward_adv(buf_len, buf_reward, buf_done, buf_value)
        if self.use_value_norm:
            self.value_normalizer.update(buf_r_sum)

        if self.share_reward:
            adv_mean = buf_advantage.mean(dim=1, keepdim=True)
            buf_advantage = adv_mean.repeat(1, self.agent_num)

        return buf_r_sum, buf_advantage

    def reward_shaping(self, buf_reward):
        # the reshaped reward is a weighted combination of the original reward and the implicit reward calculated
        # through occupancy measure matching or discriminator
        buf_reward = (1 - self.pweight) * buf_reward[:, :self.agent_num] + self.pweight * buf_reward[:, self.agent_num:]
        self.pweight *= self.pdecay
        return buf_reward

    def compute_reward_adv(self, buf_len, buf_reward, buf_done, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_sum = 0  # reward sum of previous step
        for i in reversed(range(buf_len)):
            buf_r_sum[i] = buf_reward[i] + self.gamma * (1 - buf_done[i]) * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - ((1 - buf_done) * buf_value)
        buf_advantage = (buf_advantage - buf_advantage.mean(dim=0)) / (buf_advantage.std(dim=0) + 1e-5)
        return buf_r_sum, buf_advantage

    def compute_reward_gae(self, buf_len, buf_reward, buf_done, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # advantage value
        pre_r_sum = 0  # reward sum of previous step
        pre_advantage = 0  # advantage value of previous step

        for i in reversed(range(buf_len)):
            buf_r_sum[i] = buf_reward[i] + self.gamma * (1 - buf_done[i]) * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_advantage[i] = buf_reward[i] + self.gamma * (1 - buf_done[i]) * pre_advantage - buf_value[i]
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv

        buf_advantage = (buf_advantage - buf_advantage.mean(dim=0)) / (buf_advantage.std(dim=0) + 1e-5)
        return buf_r_sum, buf_advantage
