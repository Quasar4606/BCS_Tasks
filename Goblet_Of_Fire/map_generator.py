#used to generate the map

import random
#for assigning initial positions

def map_loader(map_file): #used to load the map and return a 2D list
    with open(map_file, 'r') as text:
        map_version = [list(line.strip()) for line in text.readlines()]
    return map_version

def empty_pos(map_version): #used to give list of empty coordinates with no walls
    empty_pos_list = [] #this will be list of tupples containing coordinates
    for y,row in enumerate(map_version):
        for x,cell in enumerate(row):
            if cell == " " :
                empty_pos_list.append((x,y))
    return empty_pos_list

def assign_initial_positions(map_version) : #assigns random initial coordinates
    empty_pos_list = empty_pos(map_version) #gives list of empty positions
    #using random.sample to return 3 unique positions
    harry_pos, death_pos, cup_pos = random.sample(empty_pos_list, 3)
    return harry_pos, death_pos, cup_pos
