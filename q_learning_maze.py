#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q-Learning for Maze Navigation with Online LiDAR Policy Adaptation
Python 2.7 | ROS Melodic/Kinetic
"""

import rospy
import math
import numpy as np
import random
import os
import csv
import json
import sys
import threading

from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

# ---------------------------------------------
# MAZE LOADER
# ---------------------------------------------

def load_grid(csv_path):
    grid = []
    with open(csv_path, 'r') as f:
        for row in csv.reader(f):
            cleaned = [c.strip() for c in row if c.strip() != '']
            if cleaned:
                grid.append([int(c) for c in cleaned])
    return np.array(grid, dtype=float)


def load_config(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    start_cell = tuple(data['start']['cell'])
    goal_cell = tuple(data['goal']['cell'])
    goal_gz = tuple(data['goal']['gazebo'])
    resolution = float(data.get('resolution', 0.5))
    offset_x = float(data.get('offset_x', 0.0))
    offset_y = float(data.get('offset_y', 0.0))
    return start_cell, goal_cell, goal_gz, resolution, offset_x, offset_y


# ---------------------------------------------
# GRIDWORLD
# ---------------------------------------------

class GridWorld(object):
    FREE = 0.0
    GOAL = 1.0
    TRAP = -1.0
    BONUS = 0.5
    WALL = 9.0

    def __init__(self, grid, start_cell, goal_cell):
        self.grid = grid.copy()
        self.n_rows = grid.shape[0]
        self.n_cols = grid.shape[1]
        self.start_state = start_cell
        self.goal_cell = goal_cell
        self.state = self.start_state
        self.grid[self.grid == 1.0] = self.WALL
        self.grid[goal_cell] = self.GOAL

    def reset(self):
        self.state = self.start_state
        return self.state

    def is_terminal(self, state):
        value = self.grid[state]
        return value == self.GOAL or value == self.TRAP

    def get_next_state(self, state, action):
        row, col = state
        if action == 0:    # UP
            row = max(0, row - 1)
        elif action == 1:  # RIGHT
            col = min(self.n_cols - 1, col + 1)
        elif action == 2:  # DOWN
            row = min(self.n_rows - 1, row + 1)
        elif action == 3:  # LEFT
            col = max(0, col - 1)

        candidate = (row, col)
        if self.grid[candidate] == self.WALL:
            return state
        return candidate

    def step(self, action):
        next_state = self.get_next_state(self.state, action)
        if next_state == self.state:
            return self.state, -20, False

        reward = -5  
        cell_value = self.grid[next_state]

        if cell_value == self.GOAL:
            reward += 100
        elif cell_value == -1.0: # TRAP
            reward -= 45
        elif cell_value == self.BONUS:
            reward += 10
            self.grid[next_state] = self.FREE  
        self.state = next_state
        done = self.is_terminal(next_state)
        return next_state, reward, done

    def print_grid(self):
        for i in range(self.n_rows):
            row = ""
            for j in range(self.n_cols):
                if (i, j) == self.start_state:
                    row += " S "
                elif (i, j) == self.goal_cell:
                    row += " G "
                elif self.grid[i, j] == -1.0:
                    row += " T "
                elif self.grid[i, j] == self.WALL:
                    row += " W "
                elif self.grid[i, j] == self.BONUS:
                    row += " B "
                else:
                    row += " . "
            print(row)
        print("")


# ---------------------------------------------
# Q-LEARNING AGENT
# ---------------------------------------------

class QLearningAgent(object):
    def __init__(self, n_rows, n_cols,
                 learning_rate=0.1,
                 discount_factor=0.95,
                 exploration_rate=1.0,
                 min_exploration_rate=0.01,
                 exploration_decay=0.995):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros((n_rows, n_cols, 4), dtype=float)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3)
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(random.choice(best_actions))

    def update_q_value(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (target - current_q)

    def decay_exploration(self):
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )

    def print_policy(self, env):
        arrows = ['^', '>', 'v', '<']
        print("\nPOLICY\n")
        for i in range(self.n_rows):
            row = ""
            for j in range(self.n_cols):
                if (i, j) == env.goal_cell:
                    row += " G "
                elif env.grid[i, j] == -1.0:
                    row += " T "
                elif env.grid[i, j] == GridWorld.WALL:
                    row += " W "
                else:
                    best = np.argmax(self.q_table[i, j])
                    row += " " + arrows[best] + " "
            print(row)
        print("")


# ---------------------------------------------
# TRAINING (purely logical simulation loop)
# ---------------------------------------------

def train_agent(env, agent, action_pub, state_pub, episodes=10000, max_steps=200):
    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        visit_count = {}

        while not done and steps < max_steps and not rospy.is_shutdown():
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            if next_state not in visit_count:
                visit_count[next_state] = 0
            visit_count[next_state] += 1
            reward -= visit_count[next_state] * 0.5

            agent.update_q_value(state, action, reward, next_state, done)

            action_msg = String()
            action_msg.data = "episode={} state={} action={} reward={}".format(
                episode, state, action, reward)
            action_pub.publish(action_msg)

            state_msg = Int32MultiArray()
            state_msg.data = [next_state[0], next_state[1]]
            state_pub.publish(state_msg)

            state = next_state
            steps += 1

        agent.decay_exploration()
        if episode % 100 == 0:
            rospy.loginfo("Episode {} | epsilon={}".format(
               episode, round(agent.exploration_rate, 4)))
    rospy.loginfo("Training finished")


# ---------------------------------------------
# DYNAMIC ROS/GAZEBO TESTING MODULE
# ---------------------------------------------

class GazeboTester(object):
    def __init__(self, agent, env, goal_gz, resolution, offset_x, offset_y):
        self.agent = agent
        self.env = env
        self.goal_gz = goal_gz
        self.resolution = resolution
        self.offset_x = offset_x
        self.offset_y = offset_y
        
        self.current_gz = (0.0, 0.0)
        self.current_yaw = 0.0
        
        # Thread-safe container holding laser range limits
        self.lidar_lock = threading.Lock()
        self.is_obstacle_ahead = False
        self.safety_threshold = resolution * 0.85  # Proximity buffer block filter

        # Odometry Subscriber setup
        rospy.Subscriber('/odom', Odometry, self._odom_callback)
        rospy.sleep(1.0)  # stabilize message streams

        self.spawn_x = self.current_gz[0]
        self.spawn_y = self.current_gz[1]

        # LiDAR Scan Subscriber setup
        rospy.Subscriber('/scan', LaserScan, self._laser_callback)

        # Kobuki Navigation Publisher velocity engine
        self.vel_pub = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=10) 

    def _odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.current_gz = (x, y)

        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_yaw = yaw

    def _laser_callback(self, msg):
        """Processes 2D scan arrays inside a thread-locked frontal window zone."""
        with self.lidar_lock:
            # Handle variable angle distributions safely across different platforms
            # Calculate the total number of items corresponding to our desired 30-degree window (-15 to +15)
            angle_range_rad = math.radians(15.0)
            
            # Identify indices for front-left and front-right ranges
            indices = []
            for i, val in enumerate(msg.ranges):
                angle = msg.angle_min + (i * msg.angle_increment)
                # Normalize angle tracking fields between [-pi, pi]
                while angle > math.pi:  angle -= 2 * math.pi
                while angle < -math.pi: angle += 2 * math.pi
                
                if abs(angle) <= angle_range_rad:
                    indices.append(i)

            # Evaluate clear passage across our targeted cone indices
            obstacle_found = False
            for idx in indices:
                distance = msg.ranges[idx]
                if msg.range_min < distance < self.safety_threshold:
                    if not (math.isnan(distance) or math.isinf(distance)):
                        obstacle_found = True
                        break
            
            self.is_obstacle_ahead = obstacle_found

    def gz_to_cell(self, x, y):
        row = int(round(self.env.n_rows - 1 - (y - self.spawn_y + self.offset_y - self.resolution / 2.0) / self.resolution))
        col = int(round((x - self.spawn_x + self.offset_x - self.resolution / 2.0) / self.resolution))
        row = max(0, min(self.env.n_rows - 1, row))
        col = max(0, min(self.env.n_cols - 1, col))
        return (row, col)

    def move_to_cell(self, action, speed=0.2, angular_speed=1.0):
        target_yaws = {
            0: math.pi/2,   # UP
            1: 0.0,         # RIGHT
            2: -math.pi/2,  # DOWN
            3: math.pi      # LEFT
        }
        target_yaw = target_yaws[action]
        rate = rospy.Rate(10)

        # --- PHASE 1: ROTATE ---
        while not rospy.is_shutdown():
            error = target_yaw - self.current_yaw
            while error > math.pi:  error -= 2 * math.pi
            while error < -math.pi: error += 2 * math.pi

            if abs(error) < 0.05:
                break

            twist = Twist()
            twist.angular.z = angular_speed if error > 0 else -angular_speed
            self.vel_pub.publish(twist)
            rate.sleep()

        self.vel_pub.publish(Twist())
        rospy.sleep(0.2)

        # Before executing onward steps, check if an unexpected obstacle is present
        with self.lidar_lock:
            if self.is_obstacle_ahead:
                rospy.logwarn("LiDAR Flag: Path intersection obstructed before moving!")
                return False

        # --- PHASE 2: ADVANCE ---
        duration = self.resolution / speed
        start = rospy.Time.now().to_sec()
        
        while rospy.Time.now().to_sec() - start < duration and not rospy.is_shutdown():
            # Intercept movement immediately if an obstacle enters our safe area
            with self.lidar_lock:
                if self.is_obstacle_ahead:
                    rospy.logerr("Emergency Alert: Dynamic obstacle entered path boundary!")
                    self.vel_pub.publish(Twist()) # Force instant halt
                    return False

            twist = Twist()
            twist.linear.x = speed
            self.vel_pub.publish(twist)
            rate.sleep()

        self.vel_pub.publish(Twist())
        rospy.sleep(0.3)
        return True

    def run(self, max_steps=200):
        rospy.loginfo("Starting Online Adaptive Gazebo test run")
        rate = rospy.Rate(2)
        steps = 0

        while not rospy.is_shutdown() and steps < max_steps:
            state = self.gz_to_cell(self.current_gz[0], self.current_gz[1])
            action = int(np.argmax(self.agent.q_table[state]))

            rospy.loginfo("Current Cell {} -> Attempting Action {}".format(state, action))
            
            # Execute physical movement and monitor for sensor interruptions
            success = self.move_to_cell(action)

            if not success:
                # Deduce the coordinates of the state entry that caused the blockage
                blocked_state = self.env.get_next_state(state, action)
                rospy.logwarn("Online Correction: Localizing penalty at state matrix {}".format(blocked_state))
                
                # Execute in-place Temporal Difference value equation update
                r_penalty = -100.0
                current_q = self.agent.q_table[state][action]
                
                # Assume lookahead target value of the blocked cell drops
                target = r_penalty + self.agent.discount_factor * np.max(self.agent.q_table[blocked_state])
                self.agent.q_table[state][action] += self.agent.learning_rate * (target - current_q)
                
                rospy.loginfo("Q-table altered locally. Re-evaluating next step from current cell position.")
                # Loop immediately handles step calculation from identical coordinate location
                continue

            # Evaluate distance vector to the coordinate destination target
            x, y = self.current_gz
            dist = ((x - self.goal_gz[0])**2 + (y - self.goal_gz[1])**2) ** 0.5
            if dist < self.resolution * 0.75:
                rospy.loginfo("Goal reached successfully!")
                break

            steps += 1
            rate.sleep()


# ---------------------------------------------
# SAVE / LOAD
# ---------------------------------------------

def save_q_table(agent, filename="q_table.npy"):
    np.save(filename, agent.q_table)
    rospy.loginfo("Q-table saved to {}".format(filename))


def load_q_table(agent, filename="q_table.npy"):
    if os.path.exists(filename):
        loaded = np.load(filename)
        if loaded.shape == agent.q_table.shape:
            agent.q_table = loaded
            rospy.loginfo("Q-table loaded from {}".format(filename))
        else:
            rospy.logwarn("Shape mismatch: expected {}, got {}".format(agent.q_table.shape, loaded.shape))
    else:
        rospy.logwarn("Q-table file not found: {}".format(filename))


# ---------------------------------------------
# MAIN
# ---------------------------------------------

def main():
    mode = '--train'
    maze_path = os.path.expanduser('~/ros_mazes/csv/maze_01.csv')

    for i, arg in enumerate(sys.argv[1:]):
        if arg in ('--train', '--test'):
            mode = arg
        elif arg == '--maze' and i + 2 < len(sys.argv):
            maze_path = os.path.expanduser(sys.argv[sys.argv.index(arg) + 1])

    json_path = maze_path.replace('/csv/', '/maps/').replace('.csv', '.json')
    qtable_path = maze_path.replace('/csv/', '/maps/').replace('.csv', '_q_table.npy')

    rospy.init_node('q_learning_maze', anonymous=True)

    grid = load_grid(maze_path)
    start_cell, goal_cell, goal_gz, resolution, offset_x, offset_y = load_config(json_path)

    env = GridWorld(grid, start_cell, goal_cell)
    agent = QLearningAgent(env.n_rows, env.n_cols)

    if mode == '--train':
        action_pub = rospy.Publisher('/ql/action', String,         queue_size=10)
        state_pub = rospy.Publisher('/ql/state',  Int32MultiArray, queue_size=10)
        train_agent(env, agent, action_pub, state_pub)
        save_q_table(agent, qtable_path)
        env.print_grid()
        agent.print_policy(env)

    elif mode == '--test':
        load_q_table(agent, qtable_path)
        tester = GazeboTester(agent, env, goal_gz, resolution, offset_x, offset_y)
        tester.run()


if __name__ == '__main__':
    main()
