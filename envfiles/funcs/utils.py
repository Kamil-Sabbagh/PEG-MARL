import numpy as np
import os
from envs.rendering import fill_coords, point_in_rect, point_in_line, point_in_circle

colors = [
    (255, 0, 0),  # red
    (0, 255, 0),  # green
    (0, 0, 255),  # blue
    (112, 39, 195),  # purple
    (255, 255, 0)  # yellow
]


def generate_dir(env_name):
    os.makedirs(env_name, exist_ok=True)
    return env_name + "/" + env_name


def create_agent(tile_size, color):
    tile = np.zeros([tile_size, tile_size, 3])
    fill_coords(tile, point_in_circle(0.5, 0.5, 0.31), color)
    return tile


def create_floor(tile_size):
    tile = np.zeros([tile_size, tile_size, 3])
    fill_coords(tile, point_in_rect(0, 0.04, 0, 1), (100, 100, 100))
    fill_coords(tile, point_in_rect(0.96, 1, 0, 1), (100, 100, 100))
    fill_coords(tile, point_in_rect(0, 1, 0, 0.04), (100, 100, 100))
    fill_coords(tile, point_in_rect(0, 1, 0.96, 1), (100, 100, 100))
    return tile


def create_lava(tile_size):
    tile = np.zeros([tile_size, tile_size, 3])
    # Background color
    fill_coords(tile, point_in_rect(0, 1, 0, 1), (255, 128, 0))
    # Little waves
    for i in range(3):
        ylo = 0.3 + 0.2 * i
        yhi = 0.4 + 0.2 * i
        fill_coords(tile, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
        fill_coords(tile, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
        fill_coords(tile, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
        fill_coords(tile, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))
    return tile


def create_door(tile_size, status, color=colors[1]):
    # status: 0 -- open; 1 -- locked; 2 -- unlocked
    tile = np.zeros([tile_size, tile_size, 3])

    if status == 0:
        fill_coords(tile, point_in_rect(0.88, 1.00, 0.00, 1.00), color)
        fill_coords(tile, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))

    # Door frame and door
    elif status == 1:
        fill_coords(tile, point_in_rect(0.00, 1.00, 0.00, 1.00), color)
        fill_coords(tile, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(color))

        # Draw key slot
        fill_coords(tile, point_in_rect(0.52, 0.75, 0.50, 0.56), color)
    else:
        fill_coords(tile, point_in_rect(0.00, 1.00, 0.00, 1.00), color)
        fill_coords(tile, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
        fill_coords(tile, point_in_rect(0.08, 0.92, 0.08, 0.92), color)
        fill_coords(tile, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

        # Draw door handle
        fill_coords(tile, point_in_circle(cx=0.75, cy=0.50, r=0.08), color)
    return tile


def create_tile(tile_size, color):
    tile = np.zeros([tile_size, tile_size, 3])
    fill_coords(tile, point_in_rect(0, 1, 0, 1), color)
    return tile

