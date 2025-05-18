import pygame
import map_generator
import bfs
import numpy as np
import evaluate_harry
import q_learning

q_table = np.load("q_table.npy") #load q table

pygame.init() #initialize

#Defining the constants
CELL_SIZE = 30
ROWS = 10
COLUMNS = 15
WIDTH = COLUMNS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE

#colors for the game
SPACE_COLOR = (255,255,255) #white for walking space
WALL_COLOR = (0,0,0) #black for walls
HARRY_COLOR = (255,0,0) #red for harry
DEATH_COLOR = (0,0,255) #blue for death eater
CUP_COLOR = (0,255,0) #green for the cup

#actions
ACTIONS = ["UP","DOWN","LEFT","RIGHT"]
screen = pygame.display.set_mode((WIDTH,HEIGHT)) #setting up the display

pygame.display.set_caption("Goblet Of Fire Task") #Adding title

def draw_map(map_version,harry_pos,death_pos,cup_pos) :
    #loop
    for y in range(ROWS):
        for x in range(COLUMNS):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE) #creating rectangle
            #drawing walls and blank spaces
            if map_version[y][x] == "X":
                pygame.draw.rect(screen, WALL_COLOR, rect)
            else:
                pygame.draw.rect(screen, SPACE_COLOR, rect)

            #drawing harry
            if (x, y) == harry_pos:
                pygame.draw.rect(screen, HARRY_COLOR, rect)

            #drawing death eater
            elif (x, y) == death_pos:
                pygame.draw.rect(screen, DEATH_COLOR, rect)

            #drawing cup
            elif (x, y) == cup_pos:
                pygame.draw.rect(screen, CUP_COLOR, rect)


#the main function
def main() :
    map_version = map_generator.map_loader("Maps/V2.txt") #using V2 version

    clock = pygame.time.Clock()
    running = True

    while running: #running loop
        harry_pos, death_pos, cup_pos = map_generator.assign_initial_positions(map_version)
        # get their random positions
        episode_running = True
        no_of_steps = 0 #adding this to avoid getting stuck in an episode
        while episode_running:

            for event in pygame.event.get(): #closes when quit
                if event.type == pygame.QUIT:
                    running = False
                    episode_running = False

            action = evaluate_harry.best_action_fn(harry_pos,death_pos) #get action for harry
            next_harry_pos = q_learning.next_pos_fn(harry_pos,action) #get next position

            #so harry won't move in walls
            if map_version[next_harry_pos[1]][next_harry_pos[0]] == "X":
                next_harry_pos = harry_pos

            #bfs for death eater
            death_pos = bfs.death_next_move(map_version, death_pos, next_harry_pos)

            no_of_steps += 1 #increase count

            #conditions for going to next episode
            if next_harry_pos == cup_pos:
                print("Harry won.")
                episode_running = False

            elif next_harry_pos == death_pos:
                print("Harry lost.")
                episode_running = False
            elif no_of_steps > 100:
                print("Step Limit exceeded.")
                episode_running = False

            harry_pos = next_harry_pos #update

            # Draw everything
            draw_map(map_version, harry_pos, death_pos, cup_pos)

            pygame.display.flip()  # Update the display
            clock.tick(10)  # Controlling FPS
    pygame.quit()

if __name__ == "__main__" : #makes the main function run only if this file is directly run
    main()