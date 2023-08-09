from envs.gridworld import GridWorldEnv
from envs.appledoor import AppleDoorEnv
from envs.window import Window


def redraw(img):
    window.show_img(img)


def reset():
    obs = env.reset()
    redraw(obs["image"])


def step(action):
    print(action)
    obs, reward, done, info = env.step(action)
    print('step=', env.step_count, ', reward=', reward, ',  done? ', done)

    if done:
        print('done!')
        reset()
    else:
        redraw(obs["image"])


def key_handler(event):
    if event.key == 'escape':
        window.close()
        return
    if event.key == 'backspace':
        reset()
        return
    if event.key == 'left':
        actions.append(env.actions.left)
        if len(actions) == action_num:
            step(actions)
            actions.clear()
        return
    if event.key == 'right':
        actions.append(env.actions.right)
        if len(actions) == action_num:
            step(actions)
            actions.clear()
        return
    if event.key == 'up':
        actions.append(env.actions.up)
        if len(actions) == action_num:
            step(actions)
            actions.clear()
        return
    if event.key == 'down':
        actions.append(env.actions.down)
        if len(actions) == action_num:
            step(actions)
            actions.clear()
        return
    if event.key == ' ':
        actions.append(env.actions.stay)
        if len(actions) == action_num:
            step(actions)
            actions.clear()
        return

# upperLeftSquare_1a
# centerSquare_1a
# centerSquare_2a
# empty_1a
# env_name = "centerSquare6x6_1a"

# env = GridWorldEnv(env_name, dense_reward=True, visualize=True)


env_name = "appleDoor_b"
env = AppleDoorEnv(env_name, dense_reward=False, visualize=True)

actions = []
action_num = env.agent_num

window = Window('Grid World')
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
