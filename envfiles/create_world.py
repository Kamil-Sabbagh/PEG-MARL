import numpy as np
from envfiles.funcs.utils import *
from envfiles.funcs.create_lava_env import initialize_img


def create_upperLeftSquare():
    env_name = "upperLeftSquare"
    width = 10
    height = 10
    grid = np.zeros([height, width])
    lava = np.zeros([height, width])

    for i in range(height):
        for j in range(width):
            if i > 1 and j < 8:
                lava[i, j] = 1

    img = initialize_img(grid, lava)

    file_name = generate_dir(env_name)
    np.save(file_name + "_grid.npy", grid)
    np.save(file_name + "_lava.npy", lava)
    np.save(file_name + "_img.npy", img)


def create_centerSquare(size):
    env_name = "centerSquare" + str(size) + "x" + str(size)
    width = 10
    height = 10
    grid = np.zeros([height, width])
    lava = np.zeros([height, width])

    t = (width - size) / 2
    for i in range(height):
        for j in range(width):
            if t-1 < i < 10-t and t-1 < j < 10-t:
                lava[i, j] = 1

    img = initialize_img(grid, lava)

    file_name = generate_dir(env_name)
    np.save(file_name + "_grid.npy", grid)
    np.save(file_name + "_lava.npy", lava)
    np.save(file_name + "_img.npy", img)


def create_appleDoor():
    env_name = "appleDoor"
    width = 10
    height = 5
    grid = np.zeros([height, width])
    lava = np.zeros([height, width])

    grid[:2, 3] = 1
    grid[-2:, 3] = 1
    grid[-4:, 7] = 1

    img = initialize_img(grid, lava)

    file_name = generate_dir(env_name)
    np.save(file_name + "_grid.npy", grid)
    np.save(file_name + "_lava.npy", lava)
    np.save(file_name + "_img.npy", img)


if __name__ == '__main__':
    # create_upperLeftSquare()
    create_centerSquare(size=6)
    # create_centerSquare(size=8)
    # create_appleDoor()
