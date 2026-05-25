#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q-Learning GridWorld
Python 2.7 ROS Melodic/Kinetic
"""

import rospy
import numpy as np
import random
import os

from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray


# GRIDWORLD
class GridWorld(object):
    def __init__(self, n):
        self.nSize = n
        # 0.0 -> empty
        # 1.0 -> goal
        # -1.0 -> trap
        # 0.5 -> bonus
        # 9.0 -> wall
        self.grid = np.zeros((n, n), dtype=float)
        
        # GENERATE TRAPS
        for _ in range(n // 4):
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)

            while self.grid[x, y] != 0:
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)

            self.grid[x, y] = -1

        # GENERATE BONUS
        for _ in range(n // 2):
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)

            while self.grid[x, y] != 0:
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)

            self.grid[x, y] = 0.5

        # GENERATE WALLS
        for _ in range(n // 3):
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)

            while self.grid[x, y] != 0:
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)

            self.grid[x, y] = 9

        # GOAL
        gx = random.randint(0, n - 1)
        gy = random.randint(0, n - 1)

        while self.grid[gx, gy] != 0:
            gx = random.randint(0, n - 1)
            gy = random.randint(0, n - 1)

        self.grid[gx, gy] = 1.0

        # START
        sx = random.randint(0, n - 1)
        sy = random.randint(0, n - 1)

        while self.grid[sx, sy] != 0:
            sx = random.randint(0, n - 1)
            sy = random.randint(0, n - 1)

        self.start_state = (sx, sy)
        self.state = self.start_state

    # =====================================================
    def reset(self):

        self.state = self.start_state
        return self.state
    # =====================================================
    def is_terminal(self, state):

        value = self.grid[state]

        return value == 1 or value == -1
    # =====================================================
    def get_next_state(self, state, action):
        row = state[0]
        col = state[1]

        # UP
        if action == 0:
            row = max(0, row - 1)
        # RIGHT
        elif action == 1:
            col = min(self.nSize - 1, col + 1)
        # DOWN
        elif action == 2:
            row = min(self.nSize - 1, row + 1)
        # LEFT
        elif action == 3:
            col = max(0, col - 1)
        candidate = (row, col)
        # WALL COLLISION
        if self.grid[candidate] == 9:
            return state
        return candidate
    # =====================================================
    def step(self, action):
        next_state = self.get_next_state(self.state, action)

        # collision punishment
        if next_state == self.state:
            return self.state, -20, False
        reward = -5

        cell_value = self.grid[next_state]

        if cell_value == 1:
            reward += 100
        elif cell_value == -1:
            reward -= 45
        elif cell_value == 0.5:
            reward += 10

            # consume bonus
            self.grid[next_state] = 0

        self.state = next_state
        done = self.is_terminal(next_state)
        return next_state, reward, done
    # =====================================================
    def print_grid(self):
        for i in range(self.nSize):
            row = ""
            for j in range(self.nSize):
                if (i, j) == self.start_state:
                    row += " S "
                elif self.grid[i, j] == 1:
                    row += " G "
                elif self.grid[i, j] == -1:
                    row += " T "
                elif self.grid[i, j] == 9:
                    row += " # "
                elif self.grid[i, j] == 0.5:
                    row += " B "
                else:
                    row += " . "
            print(row)
        print("")


# Q-Learning Agent (ahora si lo chido)
class QLearningAgent(object):
    def __init__(self, nSize, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0):
        self.nSize = nSize
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.995

        # actions:
        # 0 up
        # 1 right
        # 2 down
        # 3 left

        self.q_table = np.zeros((nSize, nSize, 4), dtype=float)
    # =====================================================
    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3)
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(random.choice(best_actions))
    # =====================================================
    def update_q_value(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + \
                     self.discount_factor * \
                     np.max(self.q_table[next_state])
        self.q_table[state][action] += \
            self.learning_rate * (target - current_q)
    # =====================================================
    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
    # =====================================================
    def print_policy(self, env):
        arrows = ['^', '>', 'v', '<']
        print("")
        print("POLICY")
        print("")
        for i in range(self.nSize):
            row = ""
            for j in range(self.nSize):
                value = env.grid[i, j]
                if value == 1:
                    row += " G "
                elif value == -1:
                    row += " T "
                elif value == 9:
                    row += " # "
                else:
                    best = np.argmax(self.q_table[i, j])
                    row += " " + arrows[best] + " "
            print(row)
        print("")


# =========================
# entrenamiento ===========
# =========================
def train_agent(env, agent, action_pub, state_pub, episodes=5000, max_steps=100):
    rate = rospy.Rate(50)
    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        visit_count = {}
        while (not done) and \
              (steps < max_steps) and \
              (not rospy.is_shutdown()):
            action = agent.choose_action(state) # CHOOSE ACTION
            next_state, reward, done = env.step(action) # EXECUTE ACTION
            # ANTI LOOP
            if next_state not in visit_count:
                visit_count[next_state] = 0
            visit_count[next_state] += 1
            reward -= visit_count[next_state] * 2
            agent.update_q_value(state, action, reward, next_state, done ) # UPDATE Q TABLE
            # ROS PUBLISHERS
            action_msg = String()
            action_msg.data = \
                "episode={} state={} action={} reward={}".format(
                    episode,
                    state,
                    action,
                    reward
                )

            action_pub.publish(action_msg)
            state_msg = Int32MultiArray()
            state_msg.data = [
                next_state[0],
                next_state[1]
            ]
            state_pub.publish(state_msg)

            # NEXT STATE
            state = next_state
            steps += 1
            rate.sleep()

        agent.decay_exploration() # epsilon decay
        # logging
        if episode % 100 == 0:
            rospy.loginfo(
                "Episode {} | epsilon={}".format(
                    episode,
                    round(agent.exploration_rate, 4)
                )
            )
    rospy.loginfo("Training finished")


# SAVE / LOAD
def save_q_table(agent,
                 filename="q_table.npy"):
    np.save(filename, agent.q_table)
    rospy.loginfo("Q-table saved")


def load_q_table(agent,
                 filename="q_table.npy"):
    if os.path.exists(filename):
        loaded = np.load(filename)
        if loaded.shape == agent.q_table.shape:
            agent.q_table = loaded
            rospy.loginfo("Q-table loaded")
        else:
            rospy.logwarn("Shape mismatch")
    else:
        rospy.logwarn("Q-table file not found")