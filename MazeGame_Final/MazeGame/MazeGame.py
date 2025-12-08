import pygame
import sys
import numpy as np

from pf02_ import GridWorldGenerated, QLearningAgent, train_agent_on_env, select_start_with_pygame

pygame.init()

CELL_SIZE = 40 #tamaño de celdas en px
STEP_DELAY = 500 # delay en ms

COLOR_BG     = (30, 30, 30)
COLOR_GOAL   = (0, 200, 0)     # 1.0
COLOR_TRAP   = (200, 0, 0)     # -1.0
COLOR_BONUS  = (0, 150, 255)   # 0.5
COLOR_EMPTY  = (60, 60, 60)    # 0.0
COLOR_BORDER = (15, 15, 15)
COLOR_TEXT   = (255, 255, 255)

font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()


#generar mapa y cerebro
N = 8
env = GridWorldGenerated(N)
####
select_start_with_pygame(env, cell_size=CELL_SIZE)  
####
agent = QLearningAgent(nSize=N)
train_agent_on_env(env, agent)

'''
#cargar mapa y cerebro
N = 8 

env = GridWorldGenerated(N)
env.load_map("mi_mapa.npy")

agent = QLearningAgent(nSize=env.nSize)
agent.load_brain("mi_cerebro.npy")
'''

grid = env.grid
ROWS, COLS = grid.shape

WIDTH  = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-Learning GridWorld Viewer")

#LOAD SPRITEs
scale_factor = 0.8

#agente
drag_img = pygame.image.load("drag.png").convert_alpha()
new_w = int(CELL_SIZE * scale_factor)
new_h = int(CELL_SIZE * scale_factor)
drag_img = pygame.transform.smoothscale(drag_img, (new_w, new_h))

#meta
pez_img = pygame.image.load("pez.png").convert_alpha()
school=pygame.Surface((64,64), pygame.SRCALPHA)
new_w = int(CELL_SIZE * scale_factor)
new_h = int(CELL_SIZE * scale_factor)
pez_img = pygame.transform.smoothscale(pez_img, (new_w, new_h))

school.blit(pez_img, (0, 0))
school.blit(pez_img, (10, 0))
school.blit(pez_img, ( 0,10))
school.blit(pez_img, (10,10))

#trampa
enemigo_img = pygame.image.load("enemigo.png").convert_alpha()
new_w = int(CELL_SIZE * scale_factor)
new_h = int(CELL_SIZE * scale_factor)
enemigo_img = pygame.transform.smoothscale(enemigo_img, (new_w, new_h))

#bonus
obeja_img = pygame.image.load("obeja.png").convert_alpha()
new_w = int(CELL_SIZE * scale_factor)
new_h = int(CELL_SIZE * scale_factor)
dragobeja_img_img = pygame.transform.smoothscale(obeja_img, (new_w, new_h))

def get_cell_color(value: float):
    if value == 1.0:
        return COLOR_GOAL
    elif value == -1.0:
        return COLOR_TRAP
    elif value == 0.5:
        return COLOR_BONUS
    elif value == 0.0:
        return COLOR_EMPTY
    else:
        return (150, 150, 150)

def draw_cell(r, c, value):
    x = c * CELL_SIZE
    y = r * CELL_SIZE
    rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, COLOR_EMPTY, rect)

    if value == 1.0:
        screen.blit(school, (x, y))
    elif value == -1.0:
        screen.blit(enemigo_img, (x, y))
    elif value == 0.5:
        screen.blit(obeja_img, (x, y))

def greedy_action(state):
    return int(np.argmax(agent.q_table[state]))


#INITIAL
state = env.reset()
done = False
score = 0.0
paused = False

STEP_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(STEP_EVENT, STEP_DELAY)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        
        if event.type == STEP_EVENT and (not paused) and (not done):
            action = greedy_action(state)
            next_state, reward, done = env.step(action)
            score += reward
            state = next_state

            
            if done:
                paused = True
                cell_value = grid[state]
                if cell_value == 1.0:
                    print(f"Llegó a la META con score: {score}")
                elif cell_value == -1.0:
                    print(f"Cayó en una TRAMPA con score: {score}")

    #DRAW GRID
    screen.fill(COLOR_BG)

    for r in range(ROWS):
        for c in range(COLS):
            value = grid[r, c]
            color = get_cell_color(value)
            x = c * CELL_SIZE
            y = r * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            pygame.draw.rect(screen, color, rect)
            draw_cell(r, c, value)
            pygame.draw.rect(screen, COLOR_BORDER, rect, 1)

            '''
            #puntos de cada celda
            text_surface = font.render(str(value), True, COLOR_TEXT)
            text_rect = text_surface.get_rect(center=rect.center)
            screen.blit(text_surface, text_rect)
            '''

    #Dragoncito
    row, col = state
    player_x = col * CELL_SIZE + (CELL_SIZE - drag_img.get_width()) // 2
    player_y = row * CELL_SIZE + (CELL_SIZE - drag_img.get_height()) // 2
    screen.blit(drag_img, (player_x, player_y))

    # Score en pantalla
    score_text = f"Score: {score:.1f}"
    if paused and done:
        score_text += "  (FINAL)"
    score_surf = font.render(score_text, True, (255, 255, 0))
    screen.blit(score_surf, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
