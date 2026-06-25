# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
csv_to_ros_map.py -- Convert a maze CSV (0=free, 1=wall) to ROS map and Gazebo world.

The world is offset so that the start cell is centered at Gazebo origin (0, 0).
The robot spawns at the start position automatically with no x_pos/y_pos needed.

Usage:
    python ~/csv_to_ros_map.py <file.csv> [resolution] --start row col --goal row col [--dynamic obs.json]

    resolution: meters per cell (default: 0.5)
    --start row col: spawn cell in grid coordinates (required)
    --goal  row col: goal cell in grid coordinates (required)
    --dynamic obs.json: optional JSON with dynamic obstacle cells (test only, orange in Gazebo)

Output:
    ~/ros_mazes/maps/<name>.pgm + <name>.yaml   -> ROS map_server / AMCL
    ~/ros_mazes/worlds/<name>.world             -> Gazebo physical world
    ~/ros_mazes/maps/<name>.json                -> start/goal metadata

Dynamic obstacles JSON format:
    {"obstacles": [[row, col], [row, col], ...]}

Example:
    python ~/csv_to_ros_map.py ~/ros_mazes/csv/maze_07.csv 0.5 --start 6 4 --goal 6 7
    python ~/csv_to_ros_map.py ~/ros_mazes/csv/maze_07.csv 0.5 --start 6 4 --goal 6 7 --dynamic ~/ros_mazes/csv/maze_07_dynamic.json

Load in Gazebo (robot spawns at start automatically):
    export TURTLEBOT_GAZEBO_WORLD_FILE=~/ros_mazes/worlds/maze_07.world
    roslaunch turtlebot_gazebo turtlebot_world.launch

Load in ROS:
    rosrun map_server map_server ~/ros_mazes/maps/maze_07.yaml
"""

import csv
import json
import os
import sys
import struct

WALL_HEIGHT = 1.0  # meters

MAPS_DIR = os.path.expanduser('~/ros_mazes/maps')
WORLDS_DIR = os.path.expanduser('~/ros_mazes/worlds')


def read_csv(path):
    grid = []
    with open(path, 'r') as f:
        for row in csv.reader(f):
            cleaned = [c.strip() for c in row if c.strip() != '']
            if cleaned:
                grid.append([int(c) for c in cleaned])
    if not grid:
        raise ValueError("CSV is empty or malformed.")
    width = len(grid[0])
    for i, row in enumerate(grid):
        if len(row) != width:
            raise ValueError("Row {} has {} cells, expected {}.".format(i, len(row), width))
    return grid

def load_dynamic_obstacles(path):
    """Load dynamic obstacle cells from a JSON file.
    Expected format: {"obstacles": [[row, col], [row, col], ...]}
    Returns a list of (row, col) tuples.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return [tuple(cell) for cell in data.get('obstacles', [])]

def cell_to_gazebo(row, col, grid_height, resolution, offset_x, offset_y):
    """Convert grid (row, col) to Gazebo (x, y) with offset so start = (0, 0)."""
    x = round(col * resolution + resolution / 2.0 - offset_x, 4)
    # Invert Y: CSV row 0 is top, Gazebo Y grows upward
    y = round((grid_height - row - 1) * resolution + resolution / 2.0 - offset_y, 4)
    return x, y


def validate_cell(grid, row, col, label):
    h = len(grid)
    w = len(grid[0])
    if not (0 <= row < h and 0 <= col < w):
        raise ValueError("{} ({},{}) is out of bounds (grid is {}x{}).".format(label, row, col, h, w))
    if grid[row][col] == 1:
        raise ValueError("{} ({},{}) is a wall cell.".format(label, row, col))


def write_pgm(grid, path):
    h, w = len(grid), len(grid[0])
    with open(path, 'wb') as f:
        # PGM header: format, dimensions, max value
        f.write('P5\n{} {}\n255\n'.format(w, h).encode('ascii'))
        for row in grid:
            for cell in row:
                # 1=wall -> black (0), 0=free -> white (254)
                f.write(struct.pack('B', 0 if cell == 1 else 254))
    print("  [OK] PGM: {}".format(path))


def write_yaml(pgm_path, yaml_path, resolution):
    with open(yaml_path, 'w') as f:
        f.write(
            "image: {}\n"
            "resolution: {}\n"
            "origin: [0.0, 0.0, 0.0]\n"
            "negate: 0\n"
            "occupied_thresh: 0.65\n"
            "free_thresh: 0.196\n".format(pgm_path, resolution)
        )
    print("  [OK] YAML: {}".format(yaml_path))


def write_json(json_path, resolution, offset_x, offset_y, start_gz, goal_gz, start_cell, goal_cell):
    data = {
        "resolution": resolution,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "start": {"cell": list(start_cell), "gazebo": list(start_gz)},
        "goal":  {"cell": list(goal_cell),  "gazebo": list(goal_gz)}
    }
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("  [OK] JSON: {}".format(json_path))


def write_world(grid, world_path, resolution, offset_x, offset_y, dynamic_obstacles):
    """Generate Gazebo .world file.
    Static walls: red/dark (0.6 0.15 0.15)
    Dynamic obstacles: orange (0.9 0.5 0.1) -- physically present but meant to be detected by LiDAR
    Dynamic obstacles are NOT in the CSV, only placed here for --test worlds.
    """
    if dynamic_obstacles is None:
        dynamic_obstacles = []

    h = len(grid)
    models = ''
    wall_count = 0
    dyn_count = 0

    # --- Static walls from CSV ---
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == 1:
                x, y = cell_to_gazebo(r, c, h, resolution, offset_x, offset_y)
                z = round(WALL_HEIGHT / 2.0, 4)
                s = round(resolution, 4)
                models += """
    <model name='wall_{id}'>
      <static>1</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name='link'>
        <collision name='col'><geometry><box><size>{s} {s} {wh}</size></box></geometry></collision>
        <visual name='vis'>
          <geometry><box><size>{s} {s} {wh}</size></box></geometry>
          <material>
            <ambient>0.6 0.15 0.15 1.0</ambient>
            <diffuse>0.6 0.15 0.15 1.0</diffuse>
            <specular>0.1 0.1 0.1 1.0</specular>
          </material>
        </visual>
      </link>
    </model>""".format(id=wall_count, x=x, y=y, z=z, s=s, wh=WALL_HEIGHT)
                wall_count += 1

    # --- Dynamic obstacles: orange boxes, same size, physically solid for LiDAR ---
    for (r, c) in dynamic_obstacles:
        x, y = cell_to_gazebo(r, c, h, resolution, offset_x, offset_y)
        z = round(WALL_HEIGHT / 2.0, 4)
        s = round(resolution, 4)
        models += """
    <model name='dynamic_obs_{id}'>
      <static>1</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name='link'>
        <collision name='col'><geometry><box><size>{s} {s} {wh}</size></box></geometry></collision>
        <visual name='vis'>
          <geometry><box><size>{s} {s} {wh}</size></box></geometry>
          <material>
            <ambient>0.9 0.5 0.1 1.0</ambient>
            <diffuse>0.9 0.5 0.1 1.0</diffuse>
            <specular>0.2 0.2 0.1 1.0</specular>
          </material>
        </visual>
      </link>
    </model>""".format(id=dyn_count, x=x, y=y, z=z, s=s, wh=WALL_HEIGHT)
        dyn_count += 1

    with open(world_path, 'w') as f:
        f.write("""<?xml version="1.0" ?>
<sdf version="1.5">
  <world name="default">
    <!-- Standard sun and ground plane, required for depth sensor ray casting -->
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
    <physics type="ode">
      <real_time_update_rate>1000.0</real_time_update_rate>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>
{models}
  </world>
</sdf>
""".format(models=models))
    print(" [OK] World: {} ({} walls, {} dynamic obstacles)".format(world_path, wall_count, dyn_count))



def parse_args(argv):
    """Parse positional and optional arguments manually (no argparse for Python 2.7)."""
    csv_path = None
    resolution = 0.5
    start_cell = None
    goal_cell = None
    dynamic_path = None

    i = 1
    while i < len(argv):
        if argv[i] == '--start':
            start_cell = (int(argv[i+1]), int(argv[i+2]))
            i += 3
        elif argv[i] == '--goal':
            goal_cell = (int(argv[i+1]), int(argv[i+2]))
            i += 3
        elif argv[i] == '--dynamic':
            dynamic_path = os.path.expanduser(argv[i+1])
            i += 2
        elif csv_path is None:
            csv_path = argv[i]
            i += 1
        else:
            resolution = float(argv[i])
            i += 1

    return csv_path, resolution, start_cell, goal_cell, dynamic_path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path, resolution, start_cell, goal_cell, dynamic_path = parse_args(sys.argv)
    csv_path = os.path.expanduser(csv_path)

    if not os.path.isfile(csv_path):
        print("Error: file not found: {}".format(csv_path))
        sys.exit(1)

    if start_cell is None or goal_cell is None:
        print("Error: --start and --goal are required.")
        sys.exit(1)

    if not os.path.exists(MAPS_DIR): os.makedirs(MAPS_DIR)
    if not os.path.exists(WORLDS_DIR): os.makedirs(WORLDS_DIR)

    name = os.path.splitext(os.path.basename(csv_path))[0]
    pgm_path = os.path.join(MAPS_DIR, name + '.pgm')
    yaml_path = os.path.join(MAPS_DIR, name + '.yaml')
    world_path = os.path.join(WORLDS_DIR, name + '.world')
    json_path = os.path.join(MAPS_DIR, name + '.json')

    print("\n[1/3] Reading CSV: {}".format(csv_path))
    grid = read_csv(csv_path)
    h = len(grid)
    print("      Grid: {}x{} cells | resolution: {} m/cell".format(len(grid[0]), h, resolution))

    validate_cell(grid, start_cell[0], start_cell[1], "Start")
    validate_cell(grid, goal_cell[0], goal_cell[1], "Goal")

    # Compute offset so start cell center lands at Gazebo (0, 0)
    offset_x = start_cell[1] * resolution + resolution / 2.0
    offset_y = (h - start_cell[0] - 1) * resolution + resolution / 2.0

    # Load dynamic obstacles if provided
    dynamic_obstacles = []
    if dynamic_path:
        if not os.path.isfile(dynamic_path):
            print("Error: dynamic obstacles file not found: {}".format(dynamic_path))
            sys.exit(1)
        dynamic_obstacles = load_dynamic_obstacles(dynamic_path)
        print("      Dynamic obstacles: {} cells loaded from {}".format(len(dynamic_obstacles), dynamic_path))

    print("\n[2/3] Generating ROS map (PGM + YAML)...")
    write_pgm(grid, pgm_path)
    write_yaml(pgm_path, yaml_path, resolution)

    print("\n[3/3] Generating Gazebo world...")
    write_world(grid, world_path, resolution, offset_x, offset_y, dynamic_obstacles)

    goal_gz = cell_to_gazebo(goal_cell[0], goal_cell[1], h, resolution, offset_x, offset_y)
    write_json(json_path, resolution, offset_x, offset_y, (0.0, 0.0), goal_gz, start_cell, goal_cell)

    print("\n  Start cell {} -> Gazebo (0.0, 0.0)  [robot spawns here automatically]".format(start_cell))
    print("  Goal  cell {} -> Gazebo ({}, {})".format(goal_cell, goal_gz[0], goal_gz[1]))
    print("\n  Launch:")
    print("  export TURTLEBOT_GAZEBO_WORLD_FILE={}".format(world_path))
    print("  roslaunch turtlebot_gazebo turtlebot_world.launch")
    print("\nDone.")


if __name__ == '__main__':
    main()
