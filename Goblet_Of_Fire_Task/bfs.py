#bfs for death eater
from collections import deque
#will use collections.deque for it

def valid_neighbours_fn(pos,map_version) :
    #gives a list of valid neighbour coordinates(the one's which are not walls)
    x , y = pos
    valid_neighbours = [] #take an empty list first

    for a,b in [(0,-1),(0,1),(-1,0),(1,0)]: #denotes up,down,left,right
        x2 = x + a
        y2 = y + b

        if 0 <= x2 < 15 and 0 <= y2 < 10 and map_version[y2][x2] != "X":  #there are 15 columns and 10 rows
            valid_neighbours.append((x2,y2))
    return valid_neighbours

def bfs_implementation(map_version,start,end):
    #bfs algo for finding harry
    queue = deque([start]) #queue with start position in it
    visited = set([start]) #set of visited cells
    parent = {}  #dictionary to get parent of each cell

    while queue :
        current = queue.popleft() #code for getting current node

        if current == end : #harry found

            path = [] #Backtracking to find the path
            while current != start :
                path.append(current)
                current = parent[current]
            path.reverse() #as the path is in reverse order in list
            return path

        for valid_neighbour in valid_neighbours_fn(current,map_version): #looking at the neighbours
            if valid_neighbour not in visited :
                visited.add(valid_neighbour)
                parent[valid_neighbour] = current #storing the parent of valid_neighbour
                queue.append(valid_neighbour) #adding it to queue

    return [] #if no path found

def death_next_move(map_version,death_pos,harry_pos): #function for next move of death eater
    path = bfs_implementation(map_version,death_pos,harry_pos) #findiing the path

    if path:
        return path[0] #if path exists return this
    else:
        return death_pos  #if no path