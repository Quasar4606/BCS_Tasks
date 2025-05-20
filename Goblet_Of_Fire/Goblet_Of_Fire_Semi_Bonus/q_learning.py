import map_generator
import bfs
import numpy as np
import matplotlib.pyplot as plt
import random

#define the parameters
ALPHA = 0.25 #learning rate
ALPHA_DECAY = 0.995 #decay constant for alpha
MIN_ALPHA = 0.01 #minimum alpha
GAMMA = 0.88 #discount factor
EPSILON = 1.0 #probabilty of choosing random action vs best action
EPSILON_DECAY = 0.9997 #decay constant for epsilon
MIN_EPSILON = 0.01 #minimun of epsilon
NO_OF_EPISODES = 100000 #total no. of episodes
MAX_STEPS = 150 #maximum steps

#define the rewards
REWARD_WIN = 950 #high bonus for finding cup
REWARD_DEATH = -250 #high penalty for getting caught by death eater
REWARD_MOVEMENT = -0.1  #low penalty for movement so harry will find cup fast
REWARD_CLOSER_TO_CUP = 5  # Bonus for moving closer to cup
REWARD_AWAY_FROM_DEATH = 20  # Bonus for moving away from Death Eater
DANGER_ZONE_PENALTY = -10  #for being near death eater

#some constants
ROWS = 10
COLUMNS = 15
ACTIONS = ["UP","DOWN","LEFT","RIGHT"]  #list of possible actions

#q table - 7D: Harry_y, Harry_x, Death_y, Death_x,Cup_y,Cup_x Actions
q_table = np.zeros((ROWS, COLUMNS, ROWS, COLUMNS,ROWS,COLUMNS, 4))
#q_table = np.load("q_table.npy")

def calculate_distance(pos1, pos2):
    #Manhattan distance between two points.
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def reward_fn(harry_pos,death_pos,cup_pos,next_harry_pos): #reward function
    #rewards are defined above
    if next_harry_pos == cup_pos:
        return REWARD_WIN
    elif next_harry_pos == death_pos:
        return REWARD_DEATH
    else:
        old_dist_cup = calculate_distance(harry_pos, cup_pos)
        old_dist_death = calculate_distance(harry_pos, death_pos)
        new_dist_cup = calculate_distance(next_harry_pos, cup_pos)
        new_dist_death = calculate_distance(next_harry_pos, death_pos)

        # Danger zone penalty
        danger_penalty = DANGER_ZONE_PENALTY if new_dist_death <= 2 else 0

        # Reward for moving closer to cup
        cup_reward = (old_dist_cup - new_dist_cup) * REWARD_CLOSER_TO_CUP
        # Reward for moving away from Death Eater
        death_reward = (new_dist_death - old_dist_death) * REWARD_AWAY_FROM_DEATH
        if calculate_distance(harry_pos,cup_pos) == 1:
            cup_reward = -65  #penalty for not  taking cup even if adjacent

        if next_harry_pos == harry_pos:  #penalty to make it not stay in same cell
            return REWARD_MOVEMENT + cup_reward + death_reward -5 + danger_penalty

        return REWARD_MOVEMENT + cup_reward + death_reward + danger_penalty

def action_fn(harry_pos, death_pos,cup_pos):  #action function
    if np.random.rand() < EPSILON :
    #if generated float less than epsilon then a random action is executed
        return random.choice(ACTIONS)

    else:
    #else then action with highest q value is chosen
        x, y = harry_pos
        dx, dy = death_pos
        cx,cy = cup_pos
        return ACTIONS[np.argmax(q_table[y][x][dy][dx][cy][cx])]

def next_pos_fn(pos,action): #next state function which determines coordinates after action is executed
    x,y = pos
    #changes the coordinates according to action
    if action == "UP" and y > 0:
        y -= 1
    elif action == "DOWN" and y < ROWS - 1:
        y += 1
    elif action == "LEFT" and x > 0:
        x -= 1
    elif action == "RIGHT" and x < COLUMNS - 1:
        x += 1
    #returns modified coordinates
    return x,y

def train_harry(map_version):

    reward_per_episode = [] #will be used for plotting
    success_per_episode = [] #will store whether harry was successful or not(1 for success, 0 for fail)
    visited_positions = {} #dictionary for storing visited positions which will be used to add penalty
    global EPSILON #epsilon decays so we need to keep changing it
    global q_table #same for q table
    global ALPHA #same for alpha
    #training loop
    for episode in range(NO_OF_EPISODES):
        current_alpha = max(ALPHA * (ALPHA_DECAY ** episode), MIN_ALPHA)
        total_reward = 0 #gives the total reward in each episode
        running = True #for loop
        success = 0 #tracking success
        steps = 0 #tracking steps
        #assign random coordinates per episode
        harry_pos,death_pos,cup_pos = map_generator.assign_initial_positions(map_version)

        while running and steps <MAX_STEPS:

            action = action_fn(harry_pos, death_pos,cup_pos) #gets the action(epsilon greedy policy)
            next_harry_pos = next_pos_fn(harry_pos,action) #gets the next position for harry

            #to make sure harry doesn't go in walls
            if map_version[next_harry_pos[1]][next_harry_pos[0]] == "X" :
                next_harry_pos = harry_pos
                total_reward -= 1 #penalty

            #bfs for death eater so it moves towards new position
            death_pos = bfs.death_next_move(map_version, death_pos, next_harry_pos)

            #5 percent chance for death eater to move twice
            if next_harry_pos != death_pos :
                if np.random.rand() < 0.05:
                    bfs.death_next_move(map_version, death_pos, next_harry_pos)

            #1 percent chance for cup to spawn teleport somewhere else
            if next_harry_pos != cup_pos:
                if np.random.rand() < 0.01:
                    empty_pos_list = map_generator.empty_pos(map_version)
                    cup_pos = random.choice(empty_pos_list)
            #calculating the reward
            reward = reward_fn(harry_pos, death_pos, cup_pos,next_harry_pos)
            total_reward += reward

            #updating q table using  the bellman equation
            old_q = q_table[harry_pos[1], harry_pos[0], death_pos[1], death_pos[0],cup_pos[1],cup_pos[0], ACTIONS.index(action)] #old q value
            new_q = np.max(q_table[next_harry_pos[1], next_harry_pos[0], death_pos[1], death_pos[0],cup_pos[1],cup_pos[0]]) #max q value for next state
            tde = reward + GAMMA * new_q - old_q #temporal difference error
            q_table[harry_pos[1], harry_pos[0], death_pos[1], death_pos[0],cup_pos[1],cup_pos[0], ACTIONS.index(action)] = old_q + current_alpha * tde #Bellman update

            #code to stop the episode after certain conditions are met
            if next_harry_pos == cup_pos:
                success = 1
                running = False
            elif next_harry_pos == death_pos:
                running = False

            harry_pos = next_harry_pos #updating the coordinates
            steps += 1  # update


            #penalty for visiting same cell more than 3 times
            visited_positions[harry_pos] = visited_positions.get(harry_pos, 0) + 1 #add count
            if visited_positions[harry_pos] > 2 :
                total_reward -= 10

            #10 percent change for double move
            if np.random.rand() < 0.1 :
                action = action_fn(harry_pos, death_pos,cup_pos)  # gets the action(epsilon greedy policy)
                next_harry_pos = next_pos_fn(harry_pos, action)  # gets the next position for harry

                # to make sure harry doesn't go in walls
                if map_version[next_harry_pos[1]][next_harry_pos[0]] == "X":
                    next_harry_pos = harry_pos
                    total_reward -= 1  # penalty

                    # calculating the reward
                    reward = reward_fn(harry_pos, death_pos, cup_pos, next_harry_pos)
                    total_reward += reward

                    # updating q table using  the bellman equation
                    old_q = q_table[
                        harry_pos[1], harry_pos[0], death_pos[1], death_pos[0],cup_pos[1],cup_pos[0], ACTIONS.index(action)]  # old q value
                    new_q = np.max(q_table[next_harry_pos[1], next_harry_pos[0], death_pos[1], death_pos[
                        0],cup_pos[1],cup_pos[0]])  # max q value for next state
                    tde = reward + GAMMA * new_q - old_q  # temporal difference error
                    q_table[harry_pos[1], harry_pos[0], death_pos[1], death_pos[0],cup_pos[1],cup_pos[0], ACTIONS.index(
                        action)] = old_q + current_alpha * tde  # Bellman update

                    # code to stop the episode after certain conditions are met
                    if next_harry_pos == cup_pos:
                        success = 1
                        running = False
                    elif next_harry_pos == death_pos:
                        running = False

                    harry_pos = next_harry_pos  # updating the coordinates
                    steps += 1  # update

                    # penalty for visiting same cell more than 3 times
                    visited_positions[harry_pos] = visited_positions.get(harry_pos, 0) + 1  # add count
                    if visited_positions[harry_pos] > 2:
                        total_reward -= 10

        EPSILON = max(EPSILON*EPSILON_DECAY,MIN_EPSILON) #updating epsilon

        reward_per_episode.append(total_reward) #will be used for plotting
        success_per_episode.append(success) #will be used for plotting

        #tracking progress every 5000 episodes
        if (episode + 1) % 5000 == 0:
            print("Episode {} \n Reward Average(every 5000 episodes): {} \n Success Rate(every 5000 episodes): {}% \n Epsilon = {} \n ".format(episode+1,sum(reward_per_episode[-5000:])/5000,sum(success_per_episode[-5000:])/50,EPSILON))

    #plotting the results
    episodes = range(1,NO_OF_EPISODES+1) #gives a list from 1 to number of episodes
    fig,axis = plt.subplots(2,1) #subplot with 2 rows and 1 column

    #Reward per episode plot
    axis[0].plot(episodes,reward_per_episode,color = "red", label = "Reward per episode")
    axis[0].set_title("Reward per episode")
    axis[0].set_xlabel("Episodes")
    axis[0].set_ylabel("Reward per episode")
    axis[0].legend()

    #Success per episode plot
    axis[1].plot(episodes, success_per_episode, color="blue", label="Success per episode")
    axis[1].set_title("Success per episode")
    axis[1].set_xlabel("Episodes")
    axis[1].set_ylabel("Success per episode")
    axis[1].legend()

    plt.tight_layout() #to prevent overlap and for better spacing
    plt.show() #for showing plot


def main():
    map_version = map_generator.map_loader("Maps/V2.txt")
    train_harry(map_version)
    np.save("q_table.npy", q_table)

if __name__ == "__main__" :  #makes the main function run only if this file is directly run
    main()
