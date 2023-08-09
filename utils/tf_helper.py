import tensorboardX
import numpy as np
import utils
np.set_printoptions(precision=2)


class tb_writer:
    def __init__(self, model_dir, agent_num, use_prior):
        self.model_dir = model_dir
        self.tb_writer = tensorboardX.SummaryWriter(model_dir)
        self.csv_file, self.csv_logger = utils.get_csv_logger(self.model_dir, mode="a")
        self.agent_num = agent_num
        self.use_prior = use_prior
        self.pweight = 0
        self.now_len = 0
        self.max_len = 5000
        self.is_full = False
        self.avg_range = 100
        self.ep_num = 0
        self.frames_num = 0
        self.ep_idxes = np.zeros(self.max_len)
        self.frames = np.zeros(self.max_len)
        self.returns = np.zeros([self.max_len, agent_num * (1 + use_prior)])

        self.gid = 0
        self.grad_norm = np.zeros([self.max_len, agent_num])
        self.policy_loss = np.zeros([self.max_len, agent_num])
        self.value_loss = np.zeros([self.max_len, agent_num])

    def empty_buffer(self):
        print("Cleaning buffer")
        self.now_len = 0

    def log_csv(self):
        print(f"Log csv, current episode number: {self.ep_num}, current buffer id: {self.now_len}")
        info = np.column_stack((self.ep_idxes, self.frames, self.returns))
        self.csv_logger.writerows(info[:self.now_len].tolist())
        self.csv_file.flush()

    def add_info(self, frames, returns, pweight=0):
        self.ep_idxes[self.now_len] = self.ep_num
        self.frames[self.now_len] = frames
        self.returns[self.now_len] = returns
        self.pweight = pweight
        self.now_len += 1
        self.ep_num += 1
        self.frames_num += frames
        # self.csv_logger.

        if self.now_len >= self.max_len:
            print(f"Extend buffer length: {self.max_len} --> {self.max_len * 2}")
            self.max_len *= 2
            self.ep_idxes = np.hstack((self.ep_idxes, np.zeros(self.ep_idxes.shape)))
            self.frames = np.hstack((self.frames, np.zeros(self.frames.shape)))
            self.returns = np.vstack((self.returns, np.zeros(self.returns.shape)))
            self.grad_norm = np.vstack((self.grad_norm, np.zeros(self.grad_norm.shape)))
            self.policy_loss = np.vstack((self.policy_loss, np.zeros(self.policy_loss.shape)))
            self.value_loss = np.vstack((self.value_loss, np.zeros(self.value_loss.shape)))

    # def add_info_batch(self, frames, returns):
    #     sid = self.now_len
    #     l = min(frames.size, self.max_len - sid)
    #
    #     self.frames[sid:sid + l] = frames[:l]
    #     self.returns[sid:sid + l] = returns[:l, ]
    #     self.now_len += l
    #
    #     if self.now_len >= self.max_len:
    #         self.now_len = 0
    #         self.is_full = True
    #     if l < frames.size:
    #         l = frames.size - l
    #         self.frames[:l] = frames[l:]
    #         self.returns[:l] = returns[l:]
    #         self.now_len = l
    #         self.is_full = True

    def add_grad_info(self, aid, ploss, vloss, grad_norm):
        self.policy_loss[self.gid, aid] = ploss
        self.value_loss[self.gid, aid] = vloss
        self.grad_norm[self.gid, aid] = grad_norm
        if aid == self.agent_num - 1:
            self.gid += 1

    def log(self, idx):
        # seems to have some issues
        if self.now_len > self.avg_range:
            mean_frames = self.frames[self.now_len-self.avg_range:self.now_len].mean()
            mean_returns = self.returns[self.now_len-self.avg_range:self.now_len].mean(axis=0)
        elif self.is_full and self.now_len < self.avg_range:
            back_len = self.avg_range - self.now_len
            mean_frames = (self.frames[:self.now_len].sum() + self.frames[-back_len:].sum()) / self.avg_range
            mean_returns = (self.returns[:self.now_len].sum(axis=0) + self.returns[-back_len:].sum(axis=0)) / self.avg_range
        else:
            mean_frames = self.frames[:self.now_len].mean()
            mean_returns = self.returns[:self.now_len].mean(axis=0)

        policy_loss_mean = self.policy_loss[:self.gid].mean(axis=0)
        value_loss_mean = self.value_loss[:self.gid].mean(axis=0)
        grad_norm_mean = self.grad_norm[:self.gid].mean(axis=0)
        self.gid = 0

        print("episode ", self.ep_num, " average frames: ", mean_frames, " average returns: ", mean_returns)

        self.tb_writer.add_scalar("frames", mean_frames, idx)
        self.tb_writer.add_scalar("ep_frames", mean_frames, self.ep_num)
        for aid in range(self.agent_num):
            self.tb_writer.add_scalar("returns_a" + str(aid), mean_returns[aid], idx)
            self.tb_writer.add_scalar("ep_returns_a" + str(aid), mean_returns[aid], self.ep_num)
            if self.use_prior:
                self.tb_writer.add_scalar("shadow_returns_a" + str(aid), mean_returns[self.agent_num + aid], idx)
                self.tb_writer.add_scalar("ep_shadow_returns_a" + str(aid), mean_returns[self.agent_num + aid],
                                          self.ep_num)
            self.tb_writer.add_scalar("policy_loss_a" + str(aid), policy_loss_mean[aid], idx)
            self.tb_writer.add_scalar("value_loss_a" + str(aid), value_loss_mean[aid], idx)
            self.tb_writer.add_scalar("grad_norm_a" + str(aid), grad_norm_mean[aid], idx)

        self.tb_writer.add_scalar("pweight", self.pweight, idx)
        self.tb_writer.add_scalar("ep_pweight", self.pweight, self.ep_num)

        return mean_returns[:self.agent_num]
