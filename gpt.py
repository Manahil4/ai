import pygame
from queue import PriorityQueue

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((500, 500))
pygame.display.set_caption("Robot Navigation Game")

# Define grid (5x5 as an example)
grid = [
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0]
]

cell_size = 100  # Size of each grid cell
start_pos = (0, 0)  # Starting position of player (row, col)
goal_pos = (4, 4)   # Goal position of player
