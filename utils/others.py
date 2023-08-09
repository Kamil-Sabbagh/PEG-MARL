import torch
import numpy as np
import matplotlib.pyplot as plt
from algos.model import ActorModel, ACModel, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_dir_name(root_dir, env_name, args):
    if args.use_prior or "POfD" in args.algo:
        if args.use_suboptimal:
            suffix = "suboptimal"
        else:
            suffix = "optimal"
        if "mpe" in args.env:
            suffix += "_" + args.mpe_demonstration
    else:
        suffix = "MAPPO"

    root_dir += "/" + suffix if "mpe" in args.env else "_" + suffix

    model_dir = root_dir + "/" + env_name

    if args.mpe_num_agents == 1:
        if args.mpe_fixed_map and args.mpe_aid >= 0:
            model_dir += "_a" + str(args.mpe_aid)
        if args.mpe_tid >= 0:
            model_dir += "_t" + str(args.mpe_tid)

    if args.dense_reward:
        model_dir += "_dense"

    # algorithm name
    model_dir += "_" + args.algo

    if "PPO" in args.algo or "POfD" in args.algo:
        model_dir += "_ep" + str(args.ppo_epoch)
        model_dir += "_nbatch" + str(args.num_mini_batch)

    if args.use_prior:
        model_dir += "_wprior"
        model_dir += "_N" + str(args.N)

    if args.clip_grad:
        print("apply clip grad.")
        model_dir += "_clipGrad"

    if args.add_noise:
        print("apply stochastic update.")
        model_dir += "_gradNoise"

    if args.use_state_norm:
        print("apply state normalization.")
        model_dir += "_statenorm"

    if args.use_value_norm:
        print("apply value normalization.")
        model_dir += "_valuenorm"

    if args.use_gae:
        print("use GAE.")
        model_dir += "_useGAE"

    if args.use_prior or args.algo == "POfD":
        model_dir += "_pw" + str(args.pweight)
        model_dir += "_pd" + str(args.pdecay)

    model_dir += "_seed" + str(args.seed)

    if args.run >= 0:
        model_dir += "_run" + str(args.run)

    return model_dir


def load_prior(env_name, use_suboptimal=True):
    if use_suboptimal:
        type_str = "_suboptimal"
    else:
        type_str = ""
    if "appleDoor" in env_name:
        env_prefix = env_name
        prior_name = "priors/" + env_prefix + type_str + "_prior"
        prior = []
        for aid in range(2):
            cur_prior = np.load(prior_name + str(aid) + ".npy")
            prior.append(cur_prior)
    else:
        env_prefix = env_name.split("_")[0]
        prior_name = "priors/" + env_prefix + type_str + "_prior"
        prior = []
        agent_num = int(env_name[-2])
        if agent_num == 2:
            prior_ids = [0, 2]
        else:
            prior_ids = list(range(agent_num))
        for aid in range(agent_num):
            temp = np.load(prior_name + str(prior_ids[aid]) + ".npy")
            cur_prior = temp
            if use_suboptimal:
                cur_prior = np.zeros([5, 10, 10])
                cur_prior[:4, :, :] = temp
            prior.append(cur_prior)
    return prior


def load_expert_trajectory(env_name, args):
    use_suboptimal = args.use_suboptimal
    if "centerSquare" in env_name:
        agent_num = int(env_name[-2])
        expert_traj = load_expert_trajectory_gridworld_lava(agent_num, use_suboptimal)
    elif "appleDoor" in env_name:
        agent_num = 2
        expert_traj = load_expert_trajectory_appledoor(env_name, agent_num, use_suboptimal)
    elif "mpe" in env_name:
        mpe_demonstration = args.mpe_demonstration
        agent_num = args.mpe_num_agents
        if "fixedMap" in env_name:
            expert_traj = load_expert_trajectory_simple_spread_fixed(agent_num, mpe_demonstration)
        else:
            expert_traj = load_expert_trajectory_simple_spread(agent_num, use_suboptimal, mpe_demonstration)
    else:
        raise ValueError("No demonstration for such environment.")
    return expert_traj


def load_expert_trajectory_gridworld_lava(agent_num, use_suboptimal=True):
    if agent_num == 2:
        prior_ids = [0, 2]
    else:
        prior_ids = list(range(agent_num))
    if use_suboptimal:
        type_str = "_suboptimal"
    else:
        type_str = ""
    expert_states = []
    expert_actions = []
    for id in prior_ids:
        states = np.genfromtxt("trajs/centerSquare6x6" + type_str + "_states{0}.csv".format(id))
        actions = np.genfromtxt("trajs/centerSquare6x6" + type_str + "_actions{0}.csv".format(id), dtype=np.int32)
        expert_states.append(states)
        expert_actions.append(actions)
    expert = {"states": expert_states, "actions": expert_actions}
    return expert


def load_expert_trajectory_appledoor(env_name, agent_num, use_suboptimal=True):
    if use_suboptimal:
        type_str = "_suboptimal"
    else:
        type_str = ""
    expert_states = []
    expert_actions = []
    for aid in range(agent_num):
        states = np.genfromtxt("trajs/" + env_name + type_str + "_states{0}.csv".format(aid))
        actions = np.genfromtxt("trajs/" + env_name + type_str + "_actions{0}.csv".format(aid), dtype=np.int32)
        expert_states.append(states)
        expert_actions.append(actions)
    expert = {"states": expert_states, "actions": expert_actions}
    return expert


def load_expert_trajectory_simple_spread(agent_num, use_suboptimal=True, mpe_demonstration=None):
    path = "trajs/"
    if mpe_demonstration is not None:
        path += "mpe_" + mpe_demonstration + "/"

    if use_suboptimal:
        type_str = "_suboptimal"
    else:
        type_str = "_best"
    expert_states = []
    expert_actions = []
    for aid in range(agent_num):
        # states = np.genfromtxt("trajs/mpe_simple_spread_random" + type_str + "_states.csv")
        # actions = np.genfromtxt("trajs/mpe_simple_spread_random" + type_str + "_actions.csv")
        states = np.genfromtxt(path + "mpe_simple_spread" + type_str + "_states.csv")
        actions = np.genfromtxt(path + "mpe_simple_spread" + type_str + "_actions.csv")
        expert_states.append(states)
        expert_actions.append(actions)
    expert = {"states": expert_states, "actions": expert_actions}
    return expert


def load_expert_trajectory_simple_spread_fixed(agent_num, mpe_demonstration=None):
    # if use_suboptimal:
    #     type_str = "_suboptimal"
    # else:
    #     type_str = "_best"
    path = "trajs/"
    if mpe_demonstration is not None:
        path += "mpe_" + mpe_demonstration + "/"
    type_str = "_best"
    expert_states = []
    expert_actions = []
    for aid in range(agent_num):
        states = np.genfromtxt(path + "mpe_simple_spread" + type_str + "_states{0}.csv".format(aid))
        actions = np.genfromtxt(path + "mpe_simple_spread" + type_str + "_actions{0}.csv".format(aid), dtype=np.int32)
        expert_states.append(states)
        expert_actions.append(actions)
    expert = {"states": expert_states, "actions": expert_actions}
    return expert


def load_models(model_dir, env, model="best", use_local_obs=False):
    acmodels = []
    if model == "best":
        status = torch.load(model_dir + "/best_status.pt", map_location=device)
    elif model == "last":
        status = torch.load(model_dir + "/last_status.pt", map_location=device)
    else:
        status = torch.load(model_dir + "/status_" + str(model) + ".pt", map_location=device)
    print(f"frames: {status['num_frames']}")

    if "PPO" or "POfD" in model_dir:
        for aid in range(env.agent_num):
            acmodels.append(ACModel(env.observation_space[aid], env.action_space[aid]))

        def select_action(state, mask=None):
            actions = [0] * env.agent_num
            for aid in range(env.agent_num):
                if use_local_obs:
                    cur_state = state[aid]
                else:
                    cur_state = state.flatten()
                dist, value = acmodels[aid](cur_state, mask)
                action = dist.sample()
                actions[aid] = action
            return actions

    else:
        raise ValueError("No such algorithm!")

    for aid in range(env.agent_num):
        acmodels[aid].load_state_dict(status["model_state"][aid])
        acmodels[aid].to(device)

    return acmodels, select_action


def load_discriminator(model_dir, env, model="best"):
    discriminarors = []
    if model == "best":
        status = torch.load(model_dir + "/best_status.pt", map_location=device)
    elif model == "last":
        status = torch.load(model_dir + "/last_status.pt", map_location=device)
    else:
        status = torch.load(model_dir + "/status_" + str(model) + ".pt", map_location=device)

    for aid in range(env.agent_num):
        if "mpe" in model_dir:
            state_dim = 10
        else:
            state_dim = int(env.observation_space[aid].shape[0] / env.agent_num)
        action_num = env.action_space[aid].n
        discriminarors.append(Discriminator(state_dim, action_num))
        discriminarors[aid].load_state_dict(status["discriminator_state"][aid])
        discriminarors[aid].to(device)

    return discriminarors
