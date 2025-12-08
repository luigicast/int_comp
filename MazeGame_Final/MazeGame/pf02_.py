# -*- coding: utf-8 -*-
import numpy as np
import random
import os
import pygame
import sys

#   GridWorld (4x4 fijo)
class GridWorld:
    def __init__(self):
        # Legend:
        #  1.0 → Goal (+100)
        # -1.0 → Trap (-50)
        #  0.5 → Bonus (+10)
        #  0.0 → Empty cell
        self.grid = np.array([
            [0, 0, 0.5, 1],   # Goal at (0, 3)
            [0, -1, 0, 0],    # Trap at (1, 1)
            [0, 0.5, 0, 0],   # Bonus at (2, 1)
            [0, 0, 0, 0]      # Start at (3, 0)
        ], dtype=float)
        self.start_state = (3, 0)
        self.state = self.start_state

    def reset(self):
        self.state = self.start_state
        return self.state

    def is_terminal(self, state):
        return self.grid[state] == 1 or self.grid[state] == -1

    def get_next_state(self, state, action):
        next_state = list(state)
        if action == 0:   # Up
            next_state[0] = max(0, state[0] - 1)
        elif action == 1: # Right
            next_state[1] = min(3, state[1] + 1)
        elif action == 2: # Down
            next_state[0] = min(3, state[0] + 1)
        elif action == 3: # Left
            next_state[1] = max(0, state[1] - 1)
        return tuple(next_state)

    def step(self, action):
        next_state = self.get_next_state(self.state, action)
        cell_value = self.grid[next_state]
        reward = -5  # step penalty

        if cell_value == 1:
            reward += 100
        elif cell_value == -1:
            reward -= 45
        elif cell_value == 0.5:
            reward += 10

        self.state = next_state
        done = self.is_terminal(next_state)
        return next_state, reward, done

    def print_grid(self):
        # Visualize the gridworld (puedes comentar esto si no lo quieres)
        for i in range(self.grid.shape[0]):
            row = ""
            for j in range(self.grid.shape[1]):
                if (i, j) == self.start_state:
                    row += "🟩 "
                elif self.grid[i, j] == 1:
                    row += "💰 "
                elif self.grid[i, j] == -1:
                    row += "💀 "
                elif self.grid[i, j] == 0.5:
                    row += "✨ "
                else:
                    row += "⬜ "
            print(row)
        print()


#   GridWorldGenerated (NxN)
class GridWorldGenerated:
    def __init__(self, n):
        self.nSize = n
        # Legend:
        #  1.0 → Goal (+100)
        # -1.0 → Trap (-50)
        #  0.5 → Bonus (+10)
        #  0.0 → Empty cell

        self.grid = np.zeros((self.nSize, self.nSize), dtype=float)

        # Trampas
        for _ in range(self.nSize // 4):
            x = random.randint(0, self.nSize - 1)
            y = random.randint(0, self.nSize - 1)
            while self.grid[x, y] != 0:
                x = random.randint(0, self.nSize - 1)
                y = random.randint(0, self.nSize - 1)
            self.grid[x, y] = -1

        # Bonus
        for _ in range(self.nSize // 2):
            x = random.randint(0, self.nSize - 1)
            y = random.randint(0, self.nSize - 1)
            while self.grid[x, y] != 0:
                x = random.randint(0, self.nSize - 1)
                y = random.randint(0, self.nSize - 1)
            self.grid[x, y] = 0.5

        # Goal
        goalX = random.randint(0, self.nSize - 1)
        goalY = random.randint(0, self.nSize - 1)
        while self.grid[goalX, goalY] != 0:
            goalX = random.randint(0, self.nSize - 1)
            goalY = random.randint(0, self.nSize - 1)
        self.grid[goalX, goalY] = 1.0

        # Start
        startX = random.randint(0, self.nSize - 1)
        startY = random.randint(0, self.nSize - 1)
        while self.grid[startX, startY] != 0:
            startX = random.randint(0, self.nSize - 1)
            startY = random.randint(0, self.nSize - 1)

        self.start_state = (startX, startY)
        self.state = self.start_state

    def reset(self):
        self.state = self.start_state
        return self.state

    def is_terminal(self, state):
        return self.grid[state] == 1 or self.grid[state] == -1

    def get_next_state(self, state, action):
        next_state = list(state)
        if action == 0:   # Up
            next_state[0] = max(0, state[0] - 1)
        elif action == 1: # Right
            next_state[1] = min(self.nSize - 1, state[1] + 1)
        elif action == 2: # Down
            next_state[0] = min(self.nSize - 1, state[0] + 1)
        elif action == 3: # Left
            next_state[1] = max(0, state[1] - 1)
        return tuple(next_state)

    def step(self, action):
        next_state = self.get_next_state(self.state, action)

        # Castigo si se queda en el mismo estado (choca pared)
        if next_state == self.state:
            return self.state, -20, False

        cell_value = self.grid[next_state]
        reward = -5  # step penalty

        if cell_value == 1:
            reward += 100
        elif cell_value == -1:
            reward -= 45
        elif cell_value == 0.5:
            reward += 10

        self.state = next_state
        done = self.is_terminal(next_state)
        return next_state, reward, done

    def set_start(self, x, y):
        self.start_state = (x, y)
        self.state = self.start_state

    # --- Persistencia ---
    def save_map(self, filename="grid_map.npy"):
        np.save(filename, self.grid)
        print(f"Mapa guardado en {filename}")

    def load_map(self, filename):
        if os.path.exists(filename):
            self.grid = np.load(filename)
            self.nSize = self.grid.shape[0]
            print(f"Mapa cargado ({self.nSize}x{self.nSize})")
        else:
            raise FileNotFoundError(f"No se encontró {filename}")

    def print_grid(self):
        # gridworld
        for i in range(self.grid.shape[0]):
            row = ""
            for j in range(self.grid.shape[1]):
                if (i, j) == self.start_state:
                    row += "🟩 "
                elif self.grid[i, j] == 1:
                    row += "💰 "
                elif self.grid[i, j] == -1:
                    row += "💀 "
                elif self.grid[i, j] == 0.5:
                    row += "✨ "
                else:
                    row += "⬜ "
            print(row)
        print()


# Q-Learning Agent (ahora si lo chido)
class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=0.1, nSize=4):

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.nSize = nSize
        self.q_table = np.zeros((self.nSize, self.nSize, 4), dtype=float)

    def choose_action(self, state, force_exploit=False):
        if (not force_exploit) and (random.uniform(0, 1) < self.exploration_rate):
            return random.randint(0, 3)  # Explore
        else:
            return int(np.argmax(self.q_table[state]))  # Exploit best known action

    def update_q_value(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )

    def print_q_values(self):
        # Print Q-values for each state (cell).
        actions = ['↑', '→', '↓', '←']
        for i in range(self.nSize):
            for j in range(self.nSize):
                q_vals = self.q_table[i, j]
                best = np.argmax(q_vals)
                q_str = " | ".join([f"{actions[a]}:{q_vals[a]:6.2f}" for a in range(4)])
                print(f"State ({i},{j}) → {q_str} | Best: {actions[best]}")
            print("-" * 70)

    # --- Persistencia ---
    def save_brain(self, filename="q_table.npy"):
        np.save(filename, self.q_table)
        print(f"Cerebro (Q-Table) guardado en {filename}")

    def load_brain(self, filename):
        if os.path.exists(filename):
            loaded_table = np.load(filename)
            if loaded_table.shape == self.q_table.shape:
                self.q_table = loaded_table
                print("Cerebro cargado exitosamente.")
            else:
                print(
                    f"Error: El cerebro cargado es de tamaño {loaded_table.shape}, "
                    f"pero el agente espera {self.q_table.shape}"
                )
        else:
            print("No se encontró archivo de cerebro.")

    def run_episode_and_print_path(self, env, max_steps=50):
        # Run one greedy episode (no exploration) and print path taken.
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        path = [state]

        while not done and steps < max_steps:
            action = int(np.argmax(self.q_table[state]))  # Always exploit
            next_state, reward, done = env.step(action)
            total_reward += reward
            path.append(next_state)
            state = next_state
            steps += 1

        print(f"\nPath taken ({len(path)} steps, total reward {total_reward}):")
        print(" → ".join([str(p) for p in path]))
        if env.grid[state] == 1:
            print("Reached the GOAL!")
        elif env.grid[state] == -1:
            print("Fell into a TRAP!")


# =========================
# entrenamiento ===========
# =========================
def get_episode_count(nSize):
    cells = nSize ** 2
    factor = 150
    return cells * factor

def get_step_count(nSize):
    manhattan = (nSize - 1) * 2
    factor = 10
    return manhattan * factor

def train_agent_on_env(env, agent, episodes=None, max_steps=None):
    #Entrena un agente en un environment sin generar plots
    if episodes is None:
        episodes = get_episode_count(env.nSize if hasattr(env, "nSize") else 4)
    if max_steps is None:
        max_steps = get_step_count(env.nSize if hasattr(env, "nSize") else 4)

    for _ in range(episodes):
        state = env.reset()
        visited_in_episode = {state}
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            # Castigo extra si se cicla
            if next_state in visited_in_episode:
                reward -= 10
            else:
                visited_in_episode.add(next_state)

            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            steps += 1

    return agent
###
#seleccionar inicio
def select_start_with_pygame(env, cell_size=40):
    pygame.init()
    font = pygame.font.SysFont(None, 28)
    small_font = pygame.font.SysFont(None, 22)

    grid = env.grid
    rows, cols = grid.shape
    width, height = cols * cell_size, rows * cell_size + 80

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Seleccionar inicio (fila, columna)")

    clock = pygame.time.Clock()

    # Campos de texto
    fila_text = ""
    col_text = ""
    active_field = "fila"  # o "col"
    message = "Escribe FILA y COLUMNA, ENTER para confirmar"

    selecting = True
    while selecting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                # Mover entre campos con TAB
                if event.key == pygame.K_TAB:
                    active_field = "col" if active_field == "fila" else "fila"

                # Borrar con BACKSPACE
                elif event.key == pygame.K_BACKSPACE:
                    if active_field == "fila":
                        fila_text = fila_text[:-1]
                    else:
                        col_text = col_text[:-1]

                # Confirmar con ENTER
                elif event.key == pygame.K_RETURN:
                    try:
                        fila = int(fila_text)
                        col = int(col_text)
                        n = env.nSize
                        if 0 <= fila < n and 0 <= col < n:
                            if env.grid[fila, col] == 1.0:
                                message = "La celda es META (1.0), elige otra."
                            elif env.grid[fila, col] == -1.0:
                                message = "La celda es TRAMPA (-1.0), elige otra."
                            else:
                                env.set_start(fila, col)
                                env.state = env.start_state
                                print(f"Start fijado en: {env.start_state}")
                                selecting = False
                        else:
                            message = "Coordenadas fuera de rango."
                    except ValueError:
                        message = "Debes escribir números enteros."

                # Escribir dígitos
                else:
                    if event.unicode.isdigit():
                        if active_field == "fila":
                            fila_text += event.unicode
                        else:
                            col_text += event.unicode

        # DIBUJAR
        screen.fill((30, 30, 30))

        # Área de UI abajo
        ui_y = cell_size + 10

        # Texto FILA
        fila_label = font.render("FILA:", True, (255, 255, 255))
        screen.blit(fila_label, (10, ui_y))

        fila_box_rect = pygame.Rect(80, ui_y, 60, 30)
        pygame.draw.rect(
            screen,
            (200, 200, 200) if active_field == "fila" else (120, 120, 120),
            fila_box_rect,
        )
        fila_surf = font.render(fila_text, True, (0, 0, 0))
        screen.blit(fila_surf, (fila_box_rect.x + 5, fila_box_rect.y + 4))

        # Texto COLUMNA
        col_label = font.render("COL:", True, (255, 255, 255))
        screen.blit(col_label, (170, ui_y))

        col_box_rect = pygame.Rect(230, ui_y, 60, 30)
        pygame.draw.rect(
            screen,
            (200, 200, 200) if active_field == "col" else (120, 120, 120),
            col_box_rect,
        )
        col_surf = font.render(col_text, True, (0, 0, 0))
        screen.blit(col_surf, (col_box_rect.x + 5, col_box_rect.y + 4))

        # Mensaje
        msg_surf = small_font.render(message, True, (255, 255, 0))
        screen.blit(msg_surf, (10, ui_y + 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.display.quit()  # cerramos esta ventana, pero dejamos pygame iniciado

# Código demo
if __name__ == "__main__":
    # Ejemplo: entrenar en un 8x8, guardar mapa y cerebro
    N = 8
    env = GridWorldGenerated(N)
    agent = QLearningAgent(nSize=N)

    print("Initial Grid Layout:")
    env.print_grid()

    train_agent_on_env(env, agent)

    print("\nTraining complete.")
    # agent.print_q_values()

    env.save_map("mi_mapa.npy")
    agent.save_brain("mi_cerebro.npy")

    agent.run_episode_and_print_path(env)
