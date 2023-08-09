import numpy as np
from envs.gridworld import GridWorldEnv
import utils


def save_to_file(data, file_path):
    try:
        with open(file_path, "ab") as handle:
            np.savetxt(handle, (data,), fmt="%s")
    except FileNotFoundError:
        with open(file_path, "wb") as handle:
            np.savetxt(handle, (data,), fmt="%s")


aid = 0
env_name = "centerSquare6x6_1a_" + str(aid)
env = GridWorldEnv(env_name, visualize=True)

model_dir = 'outputs_lava_suboptimal/centerSquare6x6_1a_' + str(aid) + '_dense_PPO_r0'
model = "best"
acmodels, select_action = utils.load_models(model_dir, env, model=model)

EPISODES = 20
STEPS = 100
RENDER = True
save_gif = False
save_traj = False

if save_gif:
    from array2gif import write_gif
    frames = []

returns = [0] * EPISODES
steps = [0] * EPISODES

for episode in range(EPISODES):
    done = False
    state = env.reset()
    lp = []
    r = []
    ret = np.zeros(env.agent_num)

    for step in range(STEPS):
        if RENDER:
            img = env.render()
            if save_gif:
                frames.append(np.moveaxis(img.copy(), 2, 0))
        action = select_action(state["vec"].flatten())

        if save_traj:
            save_to_file(state["vec"].flatten(), "trajs/centerSquare6x6_suboptimal_states" + str(aid) + ".csv")
            save_to_file([action[i].item() for i in range(len(action))],
                         "trajs/centerSquare6x6_suboptimal_actions" + str(aid) + ".csv")

        state, r_, done, i_ = env.step(action)
        ret += r_

        if done or env.window.closed:
            state = env.reset()
            break

    returns[episode] = ret
    steps[episode] = step + 1
    print("episode ", episode, ": steps", step+1, ", return", ret)

    if env.window.closed:
        break

print("Averaged episode return over ", EPISODES, " episodes: ", sum(returns)/len(returns))
print("Total number of samples: ", sum(steps))

if save_gif:
    print("Saving gif... ", end="")
    write_gif(np.array(frames), model_dir + ".gif", fps=10)
    print("Done.")
