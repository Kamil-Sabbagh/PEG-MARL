from multiprocessing import Process, Pipe
import numpy as np
import gym


def worker(conn, env, local_lambda):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            i, j = env.agent_pos
            local_lambda[i, j] += 1
            if done:
                info = local_lambda.copy()
                obs = env.reset()
                local_lambda *= 0
                local_lambda[0, 0] += 1
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.local_lambdas = []
        for i in range(len(envs)):
            self.local_lambdas.append(np.zeros([envs[0].height, envs[0].width]))
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        # for env in self.envs[1:]:
        for i in range(len(envs)):
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, self.envs[i], self.local_lambdas[i]))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return tuple(results)

    def step(self, actions):
        for local, action in zip(self.locals, actions): #[1:]
            local.send(("step", action))
        # obs, reward, done, info = self.envs[0].step(actions[0])
        # i, j = self.envs[0].agent_pos
        # self.local_lambdas[0][i, j] += 1
        # if done:
        #     info = self.local_lambdas[0].copy()
        #     obs = self.envs[0].reset()
        #     self.local_lambdas[0] *= 0
        #     self.local_lambdas[0][0, 0] += 1
        # results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        results = [local.recv() for local in self.locals]
        return results
