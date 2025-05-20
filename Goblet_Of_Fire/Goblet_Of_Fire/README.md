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
NO_OF_EPISODES = 1000000 #total no. of episodes
MAX_STEPS = 150 #maximum steps

