import numpy as np
import json
import matplotlib.pyplot as plt
from envfiles.funcs.utils import *
from envs.rendering import fill_coords, point_in_rect, point_in_line, point_in_circle


def save_json_lava(env_name, agent_poses, goal_poses, suffix=None):
    agent_num = len(agent_poses)
    json_dict = {
        "agent_num": agent_num,
        "starts": agent_poses,
        "goals": goal_poses,
        "grid_file": env_name + "_grid.npy",
        "lava_file": env_name + "_lava.npy",
        "img_file": env_name + "_img.npy"
    }
    dir_name = generate_dir(env_name)
    file_name = dir_name + "_" + str(agent_num) + "a"
    if suffix:
        file_name += suffix
    file_name += ".json"
    with open(file_name, 'w') as outfile:
        json.dump(json_dict, outfile, indent=4)


def save_json_appledoor(env_name, agent_poses, goal_poses, door_pose, suffix):
    agent_num = len(agent_poses)
    json_dict = {
        "agent_num": agent_num,
        "starts": agent_poses,
        "goals": goal_poses,
        "door": door_pose,
        "grid_file": env_name + "_grid.npy",
        "lava_file": env_name + "_lava.npy",
        "img_file": env_name + "_img.npy"
    }

    dir_name = generate_dir(env_name)
    file_name = dir_name + "_" + suffix + ".json"
    with open(file_name, 'w') as outfile:
        json.dump(json_dict, outfile, indent=4)


def update_img(img, agent_poses, goal_poses, door_pose=None, tile_size=30):
    if door_pose:
        door_tile = create_door(tile_size, 2, colors[1])
        x = door_pose[0] * tile_size
        y = door_pose[1] * tile_size
        img[x:x + tile_size, y:y + tile_size] = door_tile

    for i in range(len(goal_poses)):
        goal_tile = create_tile(tile_size, colors[i])
        x = goal_poses[i][0] * tile_size
        y = goal_poses[i][1] * tile_size
        img[x:x + tile_size, y:y + tile_size] = goal_tile

    for i in range(len(agent_poses)):
        x = agent_poses[i][0] * tile_size
        y = agent_poses[i][1] * tile_size
        agent_tile = img[x:x + tile_size, y:y + tile_size]
        fill_coords(agent_tile, point_in_circle(0.5, 0.5, 0.31), colors[i])
        img[x:x + tile_size, y:y + tile_size] = agent_tile
    return img


def save_figure(env_name, agent_poses, goal_poses, door_pose=None, suffix=None):
    file_name = generate_dir(env_name)
    img = np.load(file_name + "_img.npy")
    img = update_img(img, agent_poses, goal_poses, door_pose)
    plt.imshow(np.flip(img, axis=0))
    plt.axis('off')
    agent_num = len(agent_poses)
    if door_pose:
        plt.savefig(file_name + "_" + suffix + ".jpg", bbox_inches='tight')
    else:
        file_name = file_name + "_" + str(agent_num) + "a"
        if suffix:
            file_name += suffix
        plt.savefig(file_name + ".jpg", bbox_inches='tight')
    plt.close()


def get_setups_lava(agent_num):
    width = 10
    height = 10
    agent_poses, goal_poses = [], []
    if agent_num == 1:
        agent_poses = [[0, 0]]
        goal_poses = [[height - 1, width - 1]]
    elif agent_num == 2:
        agent_poses = [[0, 0], [height - 1, width - 1]]
        goal_poses = [[height - 1, width - 1], [0, 0]]
    elif agent_num == 3:
        agent_poses = [[0, 0], [0, width - 1], [height - 1, width - 1]]
        goal_poses = [[height - 1, width - 1], [height - 1, 0], [0, 0]]
    elif agent_num == 4:
        agent_poses = [[0, 0], [0, width - 1], [height - 1, width - 1], [height - 1, 0]]
        goal_poses = [[height - 1, width - 1], [height - 1, 0], [0, 0], [0, width - 1]]
    return agent_poses, goal_poses


if __name__ == '__main__':
    names = ["centerSquare6x6", "centerSquare8x8", "upperLeftSquare", "appleDoor"]
    env_name = "appleDoor"
    # env_name = "centerSquare6x6"
    if "Square" in env_name:
        # for agent_num in range(1, 5):
        #     agent_poses, goal_poses = get_setups_lava(agent_num)
        #     save_json_lava(env_name, agent_poses, goal_poses)
        #     save_figure(env_name, agent_poses, goal_poses)
        width = 10
        height = 10
        agent_poses = [[0, 0], [0, width - 1], [height - 1, width - 1], [height - 1, 0]]
        goal_poses = [[height - 1, width - 1], [height - 1, 0], [0, 0], [0, width - 1]]
        for i in range(4):
            save_json_lava(env_name, [agent_poses[i]], [goal_poses[i]], suffix="_"+str(i))
            save_figure(env_name, [agent_poses[i]], [goal_poses[i]], suffix="_"+str(i))

    elif "appleDoor" in env_name:
        agent_poses = [[2, 0], [2, 5]]
        # goal_poses = [[2, 6], [2, 9]]
        goal_poses = [[2, 9], [0, 7]]
        door_pose = [2, 3]
        # suffix = "a"
        suffix = "b"
        save_json_appledoor(env_name, agent_poses, goal_poses, door_pose, suffix)
        save_figure(env_name, agent_poses, goal_poses, door_pose, suffix)

