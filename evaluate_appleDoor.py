import numpy as np
from envs.appledoor import AppleDoorEnv
import utils


def save_to_file(data, file_path):
    try:
        with open(file_path, "ab") as handle:
            np.savetxt(handle, (data,), fmt="%s")
    except FileNotFoundError:
        with open(file_path, "wb") as handle:
            np.savetxt(handle, (data,), fmt="%s")


aid = 2
env_name = "appleDoor_b"
env = AppleDoorEnv(f"{env_name}_{aid}", visualize=True)

model_dir = f"outputs/outputs_appleDoor_MAPPO/{env_name}_{aid}_dense_PPO_ep4_nbatch4_seed1"
model = "best"
acmodels, select_action = utils.load_models(model_dir, env, model=model)

EPISODES = 10
STEPS = 100
RENDER = True
save_gif = False
save_traj = True

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
            save_to_file(state["vec"].flatten(), f"trajs/{env_name}_suboptimal_states{aid-1}.csv")
            save_to_file([action[i].item() for i in range(len(action))],
                         f"trajs/{env_name}_suboptimal_actions{aid-1}.csv")

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
