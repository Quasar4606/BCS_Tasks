#this file is for checking how many generations it takes for harry
#to escape death eater 10 times

import numpy as np
import map_generator
import bfs
import q_learning

q_table = np.load("q_table.npy") #load q table

#some necessary constants
ROWS, COLUMNS = 10, 15
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

def best_action_fn(pos):  #function to provide the best action. As model is no longer we dont need random actions
    x,y = pos
    return ACTIONS[np.argmax(q_table[y][x])] #action with highest q value is chosen

def run_episode(map_version): #for running an episode
    harry_pos, death_pos, cup_pos = map_generator.assign_initial_positions(map_version) #load random coordinates
    running = True
    no_of_steps = 0 #added this coz I observed harry and death eater to oscillate next to each other

    while running:
        action = best_action_fn(harry_pos) #get action
        next_harry_pos = q_learning.next_pos_fn(harry_pos,action) #get next moove
        #so harry won't go in walls
        if map_version[next_harry_pos[1]][next_harry_pos[0]] == "X":
            next_harry_pos = harry_pos
        #use bfs
        death_pos = bfs.death_next_move(map_version,death_pos,next_harry_pos)
        #if conditions met
        if next_harry_pos == cup_pos:
            return True
        elif next_harry_pos == death_pos:
            return False
        #update harry coordinates
        harry_pos = next_harry_pos
        no_of_steps += 1
        if no_of_steps > 200: #makes it not get stuck
            return False
    return False

def eval_streak(map_version,streak):
    #take these 2 as 0 initially
    consecutive_wins = 0
    no_of_generations = 0
    #loop
    while consecutive_wins < streak:
        success = run_episode(map_version) #run episode
        #condition for consecutive wins
        if success:
            consecutive_wins += 1
        else:
            consecutive_wins = 0

        no_of_generations += 1 #increase count
    #print statement
    print("Harry escaped {} times in a row after {} generations".format(streak,no_of_generations))

def main():
    map_version = map_generator.map_loader("Maps/V2.txt")
    eval_streak(map_version,10)

if __name__ == "__main__":  #makes the main function run only if this file is directly run
    main()

#Note; This was result when I ran the code
#Harry escaped 10 times in a row after 57620 generations