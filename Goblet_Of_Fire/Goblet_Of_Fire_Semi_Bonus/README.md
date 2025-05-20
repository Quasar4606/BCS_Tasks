Goblet Of Fire:-
This project implements a Q-Learning agent to train Harry Potter to reach a cup while avoiding a Death Eater. The environment is grid-based and dynamic, with the Death Eater chasing Harry using BFS.

Environment:A 10x15 grid with walls ('X'), open paths, Harry, Death Eater, and Cup.

State:Defined by the positions of Harry, Death Eater, and the Cup → total states = 10*15 * 10*15 * 10*15 = ~5 million.

Actions:["UP", "DOWN", "LEFT", "RIGHT"]

Q-Table:A 7D table — Q[harry_y][harry_x][death_y][death_x][cup_y][cup_x][action]

Policy:Epsilon-greedy with decaying ε and α.
