import json
import matplotlib.pyplot as plt
from envfiles.funcs.utils import *


def initialize_img(grid, lava, tile_size=30):
    height, width = grid.shape
    img = np.zeros([height * tile_size, width * tile_size, 3], dtype=int)
    floor_tile = create_floor(tile_size)
    wall_tile = create_tile(tile_size, (100, 100, 100))
    lava_tile = create_lava(tile_size)

    for i in range(height):
        for j in range(width):
            x = i * tile_size
            y = j * tile_size
            if grid[i, j] == 1:
                img[x:x + tile_size, y:y + tile_size] = wall_tile
            else:
                if lava[i, j] == 1:
                    img[x:x + tile_size, y:y + tile_size] = lava_tile
                else:
                    img[x:x + tile_size, y:y + tile_size] = floor_tile
    return img




