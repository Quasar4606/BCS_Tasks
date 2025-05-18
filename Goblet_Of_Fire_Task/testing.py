#tests success rate by running 10000 episodes
import map_generator
import numpy as np
import evaluate_harry
from q_learning import next_pos_fn
import bfs

q_table = np.load("q_table.npy") #load q table

ACTIONS = ["UP","DOWN","LEFT","RIGHT"] #actions

def test_success_rate(map_version, episodes=10000): #function for success rate
    wins = 0 #storing wins
    for episode in range(episodes):
        harry_pos, death_pos, cup_pos = map_generator.assign_initial_positions(map_version) #random positions
        steps = 0 #steps
        max_steps = 200  # Prevent infinite loops
        while True: #loop
            action = evaluate_harry.best_action_fn(harry_pos)  #giving action
            next_harry_pos = next_pos_fn(harry_pos, action) #giving position
            if map_version[next_harry_pos[1]][next_harry_pos[0]] == "X": #for avoiding walls
                next_harry_pos = harry_pos
            death_pos = bfs.death_next_move(map_version, death_pos, next_harry_pos) #bfs
            if next_harry_pos == cup_pos:
                wins += 1
                break
            elif next_harry_pos == death_pos:
                break
            harry_pos = next_harry_pos #update
            steps += 1 #update
            #breaking statement
            if steps > max_steps:
                break
    return wins / episodes

def main() :
    map_version = map_generator.map_loader("Maps/V2.txt")
    print("Success Rate: {}%".format(test_success_rate(map_version) * 100))

if __name__ == "__main__" :
    main()