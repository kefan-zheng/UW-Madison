import heapq

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0
    for i in range(len(from_state)):
        if from_state[i] != 0:
            # current coordinate
            x1 = (int)(i/3)
            y1 = (int)(i%3)
            # correct coordinate
            x2 = (int)((from_state[i]-1)/3)
            y2 = (int)((from_state[i]-1)%3)
            # compute distance
            distance += abs(x1-x2) + abs(y1-y2)

    return distance

def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    succ_states = []
    directions = [[-1, 0], [0, -1], [0, 1], [1, 0]]

    def isCoordValid(x, y):
        return x>=0 and x <=2 and y>=0 and y<=2
    
    for i in range(len(state)):
        # find 0
        if state[i] == 0:
            x1 = (int)(i/3)
            y1 = (int)(i%3)
            # try four direction
            for j in range(len(directions)):
                x2 = x1 + directions[j][0]
                y2 = y1 + directions[j][1]
                idx = x2*3 + y2
                # check if coordinate valid
                if isCoordValid(x2, y2) and state[idx] != 0:
                    tmp_state = state[:]
                    tmp_state[i] = state[idx]
                    tmp_state[idx] = 0
                    succ_states.append(tmp_state)
                    
                    
    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """

    # priority queue (open node)
    pq = []
    # closed node
    closed_node = []
    # visited node
    visited_state = []
    # max pq length
    max_length = 0
    # init
    g = 0
    h = get_manhattan_distance(state)
    cost = g+h
    parent_index = -1
    node = (cost, state, (g, h, parent_index))
    heapq.heappush(pq, node)
    # update visited state
    visited_state.append(state)
    # loop
    while len(pq)!=0:
        #update max pq length
        max_length = max(max_length, len(pq))
        # pop smallest state node
        cur_node = heapq.heappop(pq)
        # push into closed list
        closed_node.append(cur_node)
        # update parent index
        parent_index = len(closed_node)-1
        # get current state
        cur_state = cur_node[1]
        cur_g = cur_node[2][0]
        # check goal
        if cur_state == goal_state:
            break

        # get succ state
        succ_states = get_succ(cur_state) 
        for succ_state in succ_states:
            # update cost
            succ_g = cur_g + 1
            succ_h = get_manhattan_distance(succ_state)
            succ_cost = succ_g + succ_h
            if succ_state not in visited_state:
                # push pq
                succ_data = (succ_cost, succ_state, (succ_g, succ_h, parent_index))
                heapq.heappush(pq, succ_data)
                # update visited state
                visited_state.append(succ_state)
            else:
                min_old_g = 40
                for i in range(len(pq)):
                    if pq[i][1] == succ_state:
                        min_old_g = min(min_old_g, pq[i][2][0])
                for i in range(len(closed_node)):
                    if closed_node[i][1] == succ_state:
                        min_old_g = min(min_old_g, closed_node[i][2][0])
                if succ_g < min_old_g:
                    # push pq
                    succ_data = (succ_cost, succ_state, (succ_g, succ_h, parent_index))
                    heapq.heappush(pq, succ_data)
                    # update visited state
                    visited_state.append(succ_state)


    # reconstruct path
    state_info_list = []
    while parent_index != -1:
        node = closed_node[parent_index]
        state_info_list.insert(0, [node[1], node[2][1], node[2][0]])
        parent_index = node[2][2]


    # This is a format helper.
    # build "state_info_list", for each "state_info" in the list, it contains "current_state", "h" and "move".
    # define and compute max length
    # it can help to avoid any potential format issue.
    for state_info in state_info_list:
        current_state = state_info[0]
        h = state_info[1]
        move = state_info[2]
        print(current_state, "h={}".format(h), "moves: {}".format(move))
    print("Max queue length: {}".format(max_length))

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([2,5,1,4,0,6,7,0,3])
    print()

    print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    solve([2,5,1,4,0,6,7,0,3])
    print()
