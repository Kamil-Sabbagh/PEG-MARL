import numpy as np
import argparse
from envs.mpe.environment import MPEEnv
import utils


def save_to_file(data, file_path):
    try:
        with open(file_path, "ab") as handle:
            np.savetxt(handle, (data,), fmt="%s")
    except FileNotFoundError:
        with open(file_path, "wb") as handle:
            np.savetxt(handle, (data,), fmt="%s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--seed", default=1)
    parser.add_argument('--scenario_name', default='simple_spread')
    parser.add_argument('--mpe_num_agents', default=2, type=int, help='Number of agents.')
    parser.add_argument('--mpe_num_landmarks', default=2, type=int, help='Number of Landmarks.')
    parser.add_argument('--mpe_num_walls', default=1, type=int, help='Number of Walls in MPE')
    parser.add_argument("--use_done_func", default=False, action='store_true')
    parser.add_argument('--max_steps', default=25, type=int)
    parser.add_argument('--mpe_aid', default=-1, type=int)
    parser.add_argument('--mpe_tid', default=-1, type=int)
    parser.add_argument("--mpe_fixed_map", default=False, action='store_true')
    parser.add_argument("--mpe_fixed_landmark", default=False, action='store_true')
    parser.add_argument("--mpe_sparse_reward", default=False, action='store_true')
    parser.add_argument("--mpe_use_new_reward", default=False, action='store_true')
    parser.add_argument("--mpe_mid_sparse_reward", default=False, action='store_true')
    parser.add_argument("--mpe_not_share_reward", default=False, action='store_true')
    parser.add_argument("--mpe_reward_type", default="reach3")
    args = parser.parse_args()

    args.mpe_fixed_map = True
    # args.mpe_fixed_landmark = True
    args.mpe_mid_sparse_reward = True
    args.mpe_not_share_reward = False

    RENDER = True
    save_traj = False

    env_name = "mpe_" + args.scenario_name
    env = MPEEnv(args)

    model_dir = "outputs/outputs_mpeSimpleSpread_2a2g1w_fixedmap/optimal_2a2g_1221_wall_noise/mpeMidSparse_fixedMap_simple_spread_2a_reachAll_POfD_ep4_nbatch4_pw0.1_pd1.0_seed0_run0"
    model_name = "best"
    acmodels, select_action = utils.load_models(model_dir, env, model=model_name, use_local_obs=True)

    EPISODES = 10
    STEPS = 100

    returns = [0] * EPISODES
    steps = [0] * EPISODES

    if RENDER:
        env.render()

    for episode in range(EPISODES):
        done = False
        state = env.reset()
        lp = []
        r = []
        ret = np.zeros(env.agent_num)

        for step in range(STEPS):
            if RENDER:
                env.render()
                # time.sleep(10)
            action = select_action(state["vec"])

            if save_traj:
                if model_name != "best":
                    model_name = "suboptimal"
                save_to_file(state["vec"].flatten(), "trajs/" + env_name + "_" + model_name + "_states" + str(args.mpe_aid) + ".csv")
                save_to_file([action[i].item() for i in range(len(action))],
                             "trajs/" + env_name + "_" + model_name + "_actions" + str(args.mpe_aid) + ".csv")

            state, r_, done, i_ = env.step(action)
            ret += r_
            print("step ", step, ": ", r_)

            if done:
                state = env.reset()
                break

        returns[episode] = ret
        steps[episode] = step + 1
        print("episode ", episode, ": steps", step+1, ", return", ret)

    print("Averaged episode return over ", EPISODES, " episodes: ", sum(returns)/len(returns))
    print("Total number of samples: ", sum(steps))