import argparse
import numpy as np
import torch

from envs.gridworld import GridWorldEnv
from envs.appledoor import AppleDoorEnv
from envs.mpe.environment import MPEEnv
from algos.ppo import PPO
from algos.pofd import POfD
from algos.base import ExpBuffer
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--env", default="centerSquare6x6_1a")

# mpe parameters
parser.add_argument("--scenario_name", default="simple_spread")
parser.add_argument("--use_done_func", default=False, action='store_true')
parser.add_argument('--mpe_max_steps', default=50, type=int)
parser.add_argument('--mpe_num_agents', default=3, type=int, help='Number of agents in MPE')
parser.add_argument('--mpe_num_landmarks', default=3, type=int, help='Number of landmarks in MPE')
parser.add_argument('--mpe_num_walls', default=0, type=int, help='Number of Walls in MPE')
parser.add_argument("--mpe_aid", default=-1, type=int)
parser.add_argument("--mpe_tid", default=-1, type=int)
parser.add_argument("--mpe_fixed_map", default=False, action='store_true')
parser.add_argument("--mpe_fixed_landmark", default=False, action='store_true')
parser.add_argument("--mpe_sparse_reward", default=False, action='store_true')
parser.add_argument("--mpe_mid_sparse_reward", default=False, action='store_true')
parser.add_argument("--mpe_use_new_reward", default=False, action='store_true')
parser.add_argument("--mpe_not_share_reward", default=False, action='store_true')
parser.add_argument("--mpe_reward_type", default="reachAll")
parser.add_argument("--mpe_demonstration", default="132231")
parser.add_argument("--dense_reward", default=False, action='store_true')
parser.add_argument("--local_obs", default=False, action='store_true')

# algorithm parameters
parser.add_argument("--algo", default="PPO",
                    help="algorithm to use: POfD | PPO")
parser.add_argument("--ppo_epoch", default=4, type=int)
parser.add_argument("--num_mini_batch", default=4, type=int)
parser.add_argument("--clip_grad", default=False, action='store_true')
parser.add_argument("--add_noise", default=False, action='store_true')
parser.add_argument("--use_state_norm", default=False, action='store_true')
parser.add_argument("--use_value_norm", default=False, action='store_true')
parser.add_argument("--use_gae", default=False, action='store_true')

parser.add_argument("--use_prior", default=False, action='store_true')
parser.add_argument("--use_suboptimal", default=False, action='store_true')
parser.add_argument("--pweight", default=0.02, type=float)
parser.add_argument("--pdecay", default=1, type=float)
parser.add_argument("--N", default=100, type=int)

parser.add_argument("--frames", type=int, default=5000000)
parser.add_argument('--run', type=int, default=-1)

parser.add_argument("--save_interval", default=False, action='store_true')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_frames = args.frames

# seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# create environment
if "centerSquare" in args.env:
    root_name = "lava"
    env_name = args.env
    env = GridWorldEnv(env_name, seed=args.seed, dense_reward=args.dense_reward)
    args.local_obs = False
elif "appleDoor" in args.env:
    root_name = "appleDoor"
    env_name = args.env
    env = AppleDoorEnv(env_name, seed=args.seed, dense_reward=args.dense_reward)
    args.local_obs = False
elif "mpe" in args.env:
    root_name = "mpeSimpleSpread"
    root_name += "_" + str(args.mpe_num_agents) + "a" + str(args.mpe_num_landmarks) + "g"
    if args.mpe_num_walls > 0:
        root_name += f"{args.mpe_num_walls}w"
    if args.mpe_sparse_reward:
        env_name = "mpeSparse_"
    elif args.mpe_mid_sparse_reward:
        env_name = "mpeMidSparse_"
    else:
        env_name = "mpe_"
    if args.mpe_use_new_reward:
        env_name += "newR_"
    # env_name = "mpeSparse_" if args.mpe_sparse_reward else "mpe_"
    if args.mpe_fixed_map:
        root_name += "_fixedmap"
        env_name += "fixedMap_"
    if args.mpe_fixed_landmark:
        root_name += "_fixedlandmark"
        env_name += "fixedLandmark_"
    if args.mpe_not_share_reward:
        env_name += "decen_"
    # root_name += "_saveMore"
    env_name += args.scenario_name + "_" + str(args.mpe_num_agents) + "a"
    if args.mpe_num_agents > 1:
        env_name += "_" + args.mpe_reward_type
    env = MPEEnv(args)
    args.local_obs = True
else:
    raise ValueError("Invalid environment name: {}".format(args.env))

state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n
agent_num = env.agent_num

# setup logging directory
root_dir = "outputs/outputs_" + root_name
model_dir = utils.get_model_dir_name(root_dir, env_name, args)
print("Model save at: ", model_dir)

# setup priors
if args.use_prior:
    prior = utils.load_prior(env_name, use_suboptimal=args.use_suboptimal)
else:
    prior = None

use_expert_traj = False
# setup algorithms
if args.algo == "PPO":
    max_len = 4096
    algo = PPO(env, args, target_steps=max_len, prior=prior)
    max_len += env.max_steps
elif args.algo == "POfD":
    use_expert_traj = True
    max_len = 4096 * 2
    expert_traj = utils.load_expert_trajectory(env_name, args)
    algo = POfD(env, args, expert_traj, target_steps=max_len)
    max_len += env.max_steps
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

buffer = ExpBuffer(max_len, state_dim, action_dim, agent_num, args)
tb_writer = utils.tb_writer(model_dir, agent_num, args.use_prior)

# try to load existing models
try:
    status = torch.load(model_dir + "/best_status.pt", map_location=device)
    algo.load_status(status)
    update = status["update"]
    num_frames = status["num_frames"]
    tb_writer.ep_num = status["num_episode"]
    best_return = status["best_return"]
    if args.use_prior or use_expert_traj:
        algo.pweight = status["pweight"]
except OSError:
    update = 0
    num_frames = 0
    best_return = -999999

# start to train
while num_frames < target_frames:
    frames = algo.collect_experiences(buffer, tb_writer)
    algo.update_parameters(buffer, tb_writer)
    num_frames += frames
    avg_returns = tb_writer.log(num_frames)

    update += 1
    if update % 1 == 0:
        tb_writer.log_csv()
        tb_writer.empty_buffer()
        status = {"num_frames": num_frames, "update": update,
                  "num_episode": tb_writer.ep_num, "best_return": best_return,
                  "model_state": [acmodel.state_dict() for acmodel in algo.acmodels],
                  "optimizer_state": [optimizer.state_dict() for optimizer in algo.optimizers]}
        if args.algo == "POfD":
            status["discriminator_state"] = [discrim.state_dict() for discrim in algo.discriminators]
            status["d_optimizer_state"] = [optimizer.state_dict() for optimizer in algo.d_optimizers]
        if args.use_prior or use_expert_traj:
            status["pweight"] = algo.pweight
        torch.save(status, model_dir + "/last_status.pt")
        if args.save_interval and update % 5 == 0:
            torch.save(status, model_dir + "/status_" + str(update) + "_" + str(num_frames) + ".pt")
        if np.all(avg_returns > best_return):
            best_return = avg_returns.copy()
            torch.save(status, model_dir + "/best_status.pt")
