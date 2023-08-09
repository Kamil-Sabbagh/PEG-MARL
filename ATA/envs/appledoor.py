import os
import numpy as np
import torch
import json
import gym
from gym.utils import seeding
from enum import IntEnum
from envs.rendering import fill_coords, point_in_circle
from envfiles.create_world import create_tile, colors
from envfiles.funcs.utils import create_door

# left, right, up, down
ACTIONS = [(0, -1), (0, 1), (1, 0), (-1, 0)]
ACTIONS_NAME = ["left", "right", "up", "down", "stay"]


class AppleDoorEnv(gym.Env):

    # Enumeration of possible actions
    class Actions(IntEnum):
        left = 0
        right = 1
        up = 2
        down = 3
        stay = 4

    class DoorStatus(IntEnum):
        open = 0
        locked = 1
        unlocked = 2

    def __init__(self, env_name, seed=0, reward_local=False, dense_reward=False, visualize=False):
        self.env_name = env_name
        envfile_dir = "./envfiles/" + env_name.split("_")[0] + "/"
        if "envfiles" in os.getcwd():
            envfile_dir = env_name.split("_")[0] + "/"
        json_file = envfile_dir + env_name + ".json"
        with open(json_file) as infile:
            args = json.load(infile)
        self.grid = np.load(envfile_dir + args["grid_file"])
        self.lava = np.load(envfile_dir + args["lava_file"])
        self.img = np.load(envfile_dir + args["img_file"])
        self.height, self.width = self.grid.shape

        self.random_transition_order = True
        self.np_random, _ = seeding.np_random(seed)

        self.reward_local = reward_local

        self.n_agents = args["agent_num"]

        self.starts = np.array(args["starts"])
        self.goals = np.array(args["goals"])
        self.goal_counts = {i + 1: 0 for i in range(self.n_agents)}

        self.agents = self.starts.copy()
        self.agents_pre = self.starts.copy()

        if self.n_agents == 2:
            self.door = np.array(args["door"])
            self.door_status = AppleDoorEnv.DoorStatus.locked

        self.collide = False
        self.step_in_lava = False

        self.actions = AppleDoorEnv.Actions

        self.action_space = []
        self.observation_space = []
        for _ in range(self.n_agents):
            self.action_space.append(gym.spaces.Discrete(5))
            self.observation_space.append(gym.spaces.Box(low=0, high=self.height-1, shape=(2 * self.n_agents, ), dtype='uint8'))

        self.episode_limit = 100
        self.step_count = 0

        self.dense_reward = dense_reward
        self.visualize = visualize
        self.initialize_img()
        self.cur_img = None
        self.window = None

        self.reset()

    def reset(self):
        self.step_count = 0
        self.agents = self.starts.copy()
        self.agents_pre = self.starts.copy()
        self.goal_counts = {i+1 : 0 for i in range(self.n_agents)}
        if self.n_agents == 2:
            self.close_door()
        if self.visualize:
            self.cur_img = self.img.copy()
            self.update_img()
        return self.get_obs(), self.get_state()

    def update_door_status(self):
        if np.array_equal(self.agents[1], self.goals[1]):
            self.open_door()
        else:
            self.close_door()

    def close_door(self):
        x, y = self.door
        self.grid[x, y] = 1
        self.door_status = AppleDoorEnv.DoorStatus.locked

    def open_door(self):
        x, y = self.door
        self.grid[x, y] = 0
        self.door_status = AppleDoorEnv.DoorStatus.open

    @property
    def state(self):
        cur_state = self.agents.copy()
        return cur_state

    @property
    def done(self):
        if self.step_in_lava or np.array_equal(self.agents[0], self.goals[0]) or (self.step_count >= self.episode_limit):
            done = True
        else:
            done = False
        return done

    def _occupied_by_grid(self, i, j):
        if self.grid[i, j] == 1:
            return True
        return False

    def _occupied_by_lava(self, i, j):
        if self.lava[i, j] == 1:
            return True
        return False

    def _occupied_by_agent(self, cur_id, i, j):
        for aid in range(self.n_agents):
            if aid == cur_id:
                pass
            elif np.array_equal(self.agents[aid], [i, j]):
                self.collide = True
                return True
        return False

    def _available_actions(self, agent_pos):
        available_actions = set()
        available_actions.add(self.actions.stay)
        i, j = agent_pos

        assert (0 <= i <= self.height - 1) and (0 <= j <= self.width - 1), \
            'Invalid indices'

        if (i > 0) and not self._occupied_by_grid(i - 1, j):
            available_actions.add(self.actions.down)
        if (i < self.height - 1) and not self._occupied_by_grid(i + 1, j):
            available_actions.add(self.actions.up)
        if (j > 0) and not self._occupied_by_grid(i, j - 1):
            available_actions.add(self.actions.left)
        if (j < self.width - 1) and not self._occupied_by_grid(i, j + 1):
            available_actions.add(self.actions.right)

        return available_actions

    def _transition(self, actions):
        self.agents_pre = self.agents.copy()
        idx = [i for i in range(self.n_agents)]
        if self.random_transition_order:
            self.np_random.shuffle(idx)
        for aid in idx:
            action = actions[aid]
            if torch.is_tensor(action):
                action = action.item()
            if action not in self._available_actions(self.agents[aid]):
                pass
            else:
                i, j = self.agents[aid]
                if action == self.actions.up:
                    i += 1
                if action == self.actions.down:
                    i -= 1
                if action == self.actions.left:
                    j -= 1
                if action == self.actions.right:
                    j += 1
                if not self._occupied_by_agent(aid, i, j):
                    self.agents[aid] = [i, j]

    def step(self, actions):

        self.step_count += 1
        self._transition(actions)

        if self.n_agents == 2:
            self.update_door_status()

        if self.visualize:
            self.update_img()

        reward = self._reward()

        done = self.done

        info = {"%d_counts" % (l + 1): 0.0 for l in range(self.n_agents)}
        info["win_counts"] = 0

        reaches = 0
        for i in range(self.n_agents):
            if np.array_equal(self.agents[i], self.goals[i]):
                reaches += 1
        if reaches > 0:
            self.goal_counts[reaches] += 1

        if done:
            info.update({"%d_counts" % (l + 1): self.goal_counts[l + 1] for l in range(self.n_agents)})
            if np.array_equal(self.agents[0], self.goals[0]):
                info["win_counts"] = 1

        if self.collide:
            self.collide = False
        if self.step_in_lava:
            self.step_in_lava = False

        return reward, done, info

    def _reward(self):
        rewards = [0] * self.n_agents
        reach_goal = [False] * self.n_agents
        for aid in range(self.n_agents):
            i, j = self.agents[aid]
            if (self.goals[aid][0] == i) and (self.goals[aid][1] == j):
                reach_goal[aid] = True
            if self.collide:
                rewards[aid] = -1
            elif self._occupied_by_lava(i, j):
                rewards[aid] = 0
                self.step_in_lava = True
            else:
                rewards[aid] = -abs(self.goals[aid] - self.agents[aid]).sum() / 100 if self.dense_reward else 0
        if reach_goal[0]:
            for aid in range(self.n_agents):
                rewards[aid] = 10 - 9 * (self.step_count / self.episode_limit)

        if not self.reward_local:
            rewards = sum(rewards) / self.n_agents

        return rewards

    def initialize_img(self, tile_size=30):
        for i in range(len(self.goals)):
            goal_tile = create_tile(tile_size, colors[i])
            x = self.goals[i][0] * tile_size
            y = self.goals[i][1] * tile_size
            self.img[x:x + tile_size, y:y + tile_size] = goal_tile / 2

    def update_img(self, tile_size=30):
        for i in range(self.n_agents):
            x = self.agents_pre[i][0] * tile_size
            y = self.agents_pre[i][1] * tile_size
            self.cur_img[x:x + tile_size, y:y + tile_size] = self.img[x:x + tile_size, y:y + tile_size].copy()

        if self.n_agents == 2:
            self.update_img_door()

        for i in range(self.n_agents):
            x = self.agents[i][0] * tile_size
            y = self.agents[i][1] * tile_size
            agent_tile = self.cur_img[x:x + tile_size, y:y + tile_size]
            fill_coords(agent_tile, point_in_circle(0.5, 0.5, 0.31), colors[i])
            self.cur_img[x:x + tile_size, y:y + tile_size] = agent_tile

    def update_img_door(self, tile_size=30):
        # status: 0 -- open; 1 -- locked; 2 -- unlocked
        x, y = self.door
        x *= tile_size
        y *= tile_size
        status = self.door_status
        door_tile = create_door(tile_size, status)
        self.cur_img[x:x + tile_size, y:y + tile_size] = door_tile

    def render(self, mode="human"):
        if not self.window:
            from envs import window
            self.window = window.Window('Grid World')
            self.window.show(block=False)

        if self.visualize:
            self.update_img()
        self.window.show_img(np.flip(self.cur_img, axis=0))

        return np.flip(self.cur_img, axis=0)

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "reward_shape": self.get_reward_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_state_size(self):
        return 2 * self.n_agents

    def get_obs_size(self):
        return 2 * self.n_agents

    def get_reward_size(self):
        return (self.n_agents,) if self.reward_local else (1,)

    def get_total_actions(self):
        return 5

    def get_state(self):
        return self.state.flatten()

    def get_avail_actions(self):
        return [[1]*5 for _ in range(self.n_agents)]

    def get_obs(self):
        return [self.state.flatten() for _ in range(self.n_agents)]

    def get_stats(self):
        return {}