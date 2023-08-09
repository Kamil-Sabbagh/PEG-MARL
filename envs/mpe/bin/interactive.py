#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from envs.mpe.environment import MultiAgentEnv
from envs.mpe.policy import InteractivePolicy
import envs.mpe.scenarios as scenarios

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple_spread.py', help='Path of the scenario Python script.')
    parser.add_argument('--mpe_num_agents', default=3, type=int, help='Number of agents in MPE')
    parser.add_argument('--mpe_num_landmarks', default=3, type=int, help='Number of landmarks in MPE')
    parser.add_argument("--use_done_func", default=False, action='store_true')
    parser.add_argument('--max_steps', default=100, type=int)
    parser.add_argument("--mpe_sparse_reward", default=False, action='store_true')
    parser.add_argument('--mpe_aid', default=-1, type=int)
    parser.add_argument("--mpe_fixed_map", default=False, action='store_true')
    parser.add_argument("--mpe_fixed_landmark", default=False, action='store_true')
    parser.add_argument("--mpe_not_share_reward", default=False, action='store_true')
    parser.add_argument("--mpe_mid_sparse_reward", default=False, action='store_true')
    parser.add_argument("--mpe_reward_type", default="reach3")
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    env = MultiAgentEnv(world, args, scenario.reset_world, scenario.reward, scenario.observation, done_callback=scenario.done, info_callback=None, shared_viewer=True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env,i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n["vec"][i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
        env.render()
        # display rewards
        for agent in env.world.agents:
           print(agent.name + " reward: %0.3f" % env._get_reward(agent) + ",  done: " + str(env._get_done(agent)))
