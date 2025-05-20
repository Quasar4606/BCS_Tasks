Goblet Of Fire:-

This project implements a Q-Learning agent to train Harry Potter to reach a cup while avoiding a Death Eater. The environment is grid-based and dynamic, with the Death Eater chasing Harry using BFS.


Environment:A 10x15 grid with walls ('X'), open paths, Harry, Death Eater, and Cup.


State:Defined by the positions of Harry, Death Eater, and the Cup → total states = 10*15 * 10*15 * 10*15 = 3375000


Actions:["UP", "DOWN", "LEFT", "RIGHT"]


Q-Table:A 7D table — Q[harry_y][harry_x][death_y][death_x][cup_y][cup_x][action]


Policy:Epsilon-greedy with decaying ε and α.


Rewards:

REWARD_WIN = 950 #high bonus for finding cup

REWARD_DEATH = -250 #high penalty for getting caught by death eater

REWARD_MOVEMENT = -0.1  #low penalty for movement so harry will find cup fast

REWARD_CLOSER_TO_CUP = 5  # Bonus for moving closer to cup

REWARD_AWAY_FROM_DEATH = 20  # Bonus for moving away from Death Eater

DANGER_ZONE_PENALTY = -10  #for being near death eater

-5 for staying in same position

-65 for not taking cup even if adjacent

-1 if trying to move towards wall

-10 for moving in same cell more than 3 times


Parameters:

ALPHA = 0.25 #learning rate

ALPHA_DECAY = 0.995 #decay constant for alpha

MIN_ALPHA = 0.01 #minimum alpha

GAMMA = 0.88 #discount factor

EPSILON = 1.0 #probabilty of choosing random action vs best action

EPSILON_DECAY = 0.9997 #decay constant for epsilon

MIN_EPSILON = 0.01 #minimun of epsilon

NO_OF_EPISODES = 1000000 #total no. of episodes(per batch)

MAX_STEPS = 150 #maximum steps


Assumptions:

In pygame, I have used the following colors:

Red : Harry

Blue : Death Eater

Green : Cup

Black : Walls

White : Path

File Structure:

Maps: contains maps(I hve used V2 for this)

map_generator.py: contains functions to load maps and assign initial positions for harry,cup and death eater

bfs.py: bfs algo for death eater

q_learning.py: q learning for training harry and also graph plotting

pygame_visual.py: visual representation for pygame

testing.py: getting the success rate of model 

evaluate_harry.py: used to get number of generations for getting 10 consecutive wins

q_table_compressed.py: compressed q table

compress.py: compressing the q table obtained from q_learning.py

decompress.py: decompressing the compressed q table

Plots : contains the graphs

Basic Approach:

So in this task I have run this model for a total of 11000000 episodes in batches of 1000000 episodes each.

For the first batch I have run this code initializing a zero numpy array for the next batches I have loaded the preexisting q table to modify it.

The parameters and rewards for the first batch are above. After the second batch I changed alpha to 0.04 and epsilon to 0.1. After the 3rd batch I realsied the model is moving in same cell for more number of times so I have increaed penalty for moving in same cell from -10 to -50.

I did this as the q table size is very big(It wasn't even uploading on github so I had to compress it). Finally I could get a success rate of around 70%.


How to Run:

First you need to run decompress.py to get the q_table.npy file which contains q table.(very important)

After that if you want to check the pygame visual then you can run pygame_visual.py.

If you want to check number of generations for harry to get 10 consecutive wins then run evaluate_harry.py and if you want to check success rate then run testing.py.



