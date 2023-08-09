from glob import glob
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FuncFormatter


params = {'legend.fontsize': 28, #24, 28
          'axes.labelsize': 24,
          'axes.titlesize': 26,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'lines.linewidth': 2.5
          }
plt.rcParams.update(params)


class AsymScale(mscale.ScaleBase):
    name = 'asym'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self, axis)
        self.a = kwargs.get("a", 1)

    def get_transform(self):
        return self.AsymTrans(self.a)

    def set_default_locators_and_formatters(self, axis):
        # possibly, set a different locator and formatter here.
        fmt = lambda x,pos: "{}".format(np.abs(x))
        axis.set_major_formatter(FuncFormatter(fmt))

    class AsymTrans(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, a):
            mtransforms.Transform.__init__(self)
            self.a = a

        def transform_non_affine(self, x):
            return (x >= 0)*x + (x < 0)*x*self.a

        def inverted(self):
            return AsymScale.InvertedAsymTrans(self.a)

    class InvertedAsymTrans(AsymTrans):

        def transform_non_affine(self, x):
            return (x >= 0)*x + (x < 0)*x/self.a
        def inverted(self):
            return AsymScale.AsymTrans(self.a)


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def get_multi_runs_mean_var(ax, folder, num, frames, label, color, pick_top=False, use_label=False, pick_middle=False):
    test = False
    if not test:
        folders = glob(folder + "*/", recursive=False)
        rewards = []
        xs = np.array(range(frames))
        for f in folders:
            file_name = f + "log.csv"
            data = np.genfromtxt(file_name, delimiter=',')
            cur_rewards = np.zeros(frames)
            cur_f = 0
            if data.shape[1] - 2 == 1:
                cur_r = data[0, 2]
            else:
                cur_r = data[0, 2: 2+num].mean()
            for i in range(len(data)):
                d = data[i]
                last_f = cur_f
                cur_f += int(d[1])
                cur_rewards[last_f:cur_f] = cur_r
                if len(d) - 2 == 1:
                    cur_r = d[2]
                else:
                    cur_r = d[2: 2 + num].mean()
            cur_rewards[cur_f:] = cur_r
            rewards.append(cur_rewards)

        step = 1000
        xs = xs[range(0, frames, step)]
        rewards = np.array([rs[range(0, frames, step)] for rs in rewards])

        if pick_top:
            idx = np.argsort(rewards[:, -1].squeeze())
            rewards = rewards[idx[5:]]
        if pick_middle:
            idx = np.argsort(rewards[:, -1].squeeze())
            sid = math.floor(len(idx) * 0.25)
            eid = math.ceil(len(idx) * 0.75)
            if len(idx) == 10:
                rewards = rewards[idx[sid:eid]]
    else:
        # For test only
        xs = list(range(1, 4900000, 100))
        rewards = np.random.rand(10, len(xs)) * 10

    rmean = np.array(smooth(np.mean(rewards, axis=0), 0.99))
    rstd = 0.5 * np.array(smooth(np.std(rewards, axis=0), 0.99))

    ax.plot(xs, rmean, color=color, label=label if use_label else None)
    ax.fill_between(xs, rmean + rstd, rmean - rstd, color=color, alpha=0.1)
    ax.set_aspect("auto")


def plot_lava(ax, agent_num, mode, pick_top=False, use_label=False, pick_middle=False):
    # model = optimal or suboptimal
    frames = 4900000

    root_folder = f"./outputs/outputs_lava_{mode}/"
    folder = root_folder + "centerSquare6x6_" + str(
        agent_num) + "a_PPO_ep4_nbatch4_wprior_N100_gradNoise_pw1.0_pd0.995"
    label = "EG-MARL(B=100)"
    get_multi_runs_mean_var(ax, folder, agent_num, frames, label, "green", pick_top, use_label)

    folder = root_folder + "centerSquare6x6_" + str(agent_num) + "a_POfD_ep4_nbatch4_pw0.02_pd1.0"
    label = "GEG-MARL"
    get_multi_runs_mean_var(ax, folder, agent_num, frames, label, "orange", pick_top, use_label, pick_middle)

    root_folder = f"./outputs/outputs_lava_MAPPO/"
    folder = root_folder + "centerSquare6x6_" + str(agent_num) + "a_PPO_ep4_nbatch4"
    label = "MAPPO"
    get_multi_runs_mean_var(ax, folder, agent_num, frames, label, "blue", pick_top, use_label, pick_middle)

    root_folder = f"./outputs/outputs_lava_{mode}_gail/"
    folder = root_folder + "centerSquare6x6_" + str(agent_num) + "a_POfD_ep4_nbatch4_pw1.0_pd1.0"
    label = "MAGAIL"
    get_multi_runs_mean_var(ax, folder, agent_num, frames, label, "purple", pick_top, use_label, pick_middle)

    folder = "./outputs/ATA_results/centerSquare6x6_" + str(agent_num) + "a/ata/csv_logs/"
    label = "ATA"
    get_multi_runs_mean_var(ax, folder, agent_num, frames, label, "brown", pick_top, use_label, pick_middle)

    xs = np.array(range(frames)).tolist()
    ax.plot(xs, np.ones(len(xs)) * 8.38, '--', color='red', label="Oracle" if use_label else None)
    if agent_num == 2:
        ax.set_yscale("asym", a=1/8)
    elif agent_num == 3:
        ax.set_yscale("asym", a=1/10)
    elif agent_num == 4:
        ax.set_yscale("asym", a=1/16)
    ax.set_title(f"{agent_num} agents", style='italic')
    ax.set_xlabel("Number of Timesteps")
    ax.set_ylabel("Episode Rewards")


def plot_appleDoor(env_type, ax, mode, pick_top=False, use_label=False, pick_middle=False):
    # type = "a" or "b"
    agent_num = 2
    frames = 4900000

    root_folder = f"./outputs/outputs_appleDoor_{mode}/"
    folder = root_folder + "appleDoor_" + env_type + "_PPO_ep4_nbatch4_wprior_N100_gradNoise_pw1.0_pd0.995"
    label = "EG-MARL(B=100)"
    get_multi_runs_mean_var(ax, folder, agent_num, frames, label, "green", pick_top, use_label, pick_middle)

    folder = root_folder + "appleDoor_" + env_type + "_POfD_ep4_nbatch4_pw0.02_pd1.0"
    label = "GEG-MARL"
    get_multi_runs_mean_var(ax, folder, agent_num, frames, label, "orange", pick_top, use_label, pick_middle)

    root_folder = f"./outputs/outputs_appleDoor_MAPPO/"
    folder = root_folder + "appleDoor_" + env_type + "_PPO_ep4_nbatch4"
    label = "MAPPO"
    get_multi_runs_mean_var(ax, folder, agent_num, frames, label, "blue", pick_top, use_label, pick_middle)

    root_folder = f"./outputs/outputs_appleDoor_{mode}_gail/"
    folder = root_folder + "appleDoor_" + env_type + "_POfD_ep4_nbatch4_pw1.0_pd1.0"
    label = "MAGAIL"
    get_multi_runs_mean_var(ax, folder, agent_num, frames, label, "purple", pick_top, use_label, pick_middle)

    folder = "./outputs/ATA_results/appleDoor_" + env_type + "/ata/csv_logs/seed"
    label = "ATA"
    get_multi_runs_mean_var(ax, folder, agent_num, frames, label, "brown", pick_top, use_label, pick_middle)

    # plot the oracle
    if type == "a":
        best_return = 9.01
    else:  # type = "b"
        best_return = 8.65
    xs = np.array(range(frames)).tolist()
    ax.plot(xs, np.ones(len(xs)) * best_return, '--', color='red', label="Oracle" if use_label else None)
    if env_type == "b" and mode == "optimal":
        ax.set_yscale("asym", a=0.25)
    elif env_type == "b" and mode == "suboptimal":
        ax.set_yscale("asym", a=0.5)
    ax.set_title("case " + env_type, style='italic')
    ax.set_xlabel("Number of Timesteps")
    ax.set_ylabel("Episode Rewards")


def plot_mpe(ax):
    agent_num = 2

    frames = 19000000
    root_folder = "outputs/outputs_mpeSimpleSpread_2a2g1w_fixedmap/"
    folder = root_folder + "MAPPO/mpeMidSparse_newR_fixedMap_simple_spread_2a_reachAll_PPO_ep4_nbatch4_seed"
    label = "MAPPO"
    get_multi_runs_mean_var(ax, folder, agent_num, frames, label, "orange", pick_top=False, use_label=True, pick_middle=False)

    folder = root_folder + "optimal_2a2g_1221_wall_noise/mpeMidSparse_newR_fixedMap_simple_spread_2a_reachAll_POfD_ep4_nbatch4_pw0.1_pd1.0_seed"
    label = "GEG-MARL"
    get_multi_runs_mean_var(ax, folder, agent_num, frames, label, "blue", pick_top=False, use_label=True, pick_middle=False)

    ax.legend(loc='lower right')
    ax.set_xlabel("Number of Timesteps")
    ax.set_ylabel("Episode Rewards")


if __name__ == "__main__":
    mscale.register_scale(AsymScale)

    # plot gridworld lava results
    ns = [2, 3, 4]
    modes = ["optimal", "suboptimal"]
    pick_top_5s = [False]
    pick_middles = [False]
    for mode in modes:
        for pick_top_5 in pick_top_5s:
            for pick_middle in pick_middles:
                num = len(ns)
                fig, axs = plt.subplots(1, num, figsize=(9*num, 7.8))
                for i in range(num):
                    use_label = True if i == 0 else False
                    plot_lava(axs[i], agent_num=ns[i], mode=mode, pick_top=pick_top_5, use_label=use_label, pick_middle=pick_middle)

                fig_name = f"lava_{mode}"
                if pick_top_5:
                    fig_name += "_top5"
                if pick_middle:
                    fig_name += "_IQM"

                fig.legend(loc='upper center', ncol=6)
                fig.tight_layout()
                fig.subplots_adjust(top=0.8)

                plt.savefig(f"{fig_name}.pdf")
                plt.close()

    # plot gridworld appleDoor results
    env_types = ["a", "b"]
    modes = ["optimal", "suboptimal"]
    pick_top_5s = [False]
    pick_middles = [False]
    for mode in modes:
        for pick_top_5 in pick_top_5s:
            for pick_middle in pick_middles:
                num = len(env_types)
                fig, axs = plt.subplots(1, num, figsize=(9*num, 8.8))
                for i in range(num):
                    use_label = True if i == 0 else False
                    plot_appleDoor(env_types[i], axs[i], mode, pick_top=pick_top_5, use_label=use_label, pick_middle=pick_middle)

                fig_name = f"appleDoor_{mode}"
                if pick_top_5:
                    fig_name += "_top5"
                if pick_middle:
                    fig_name += "_IQM"

                fig.legend(loc='upper center', ncol=3)
                fig.tight_layout()
                fig.subplots_adjust(top=0.78)

                plt.savefig(f"{fig_name}.pdf")
                plt.close()

    # plot mpe results
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    plot_mpe(ax)
    fig.tight_layout()
    plt.savefig("mpe.pdf")
    plt.close()
