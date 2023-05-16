import math
import time
from copy import deepcopy
from heapq import heappush, heappop

class Problem:
    def __init__(self, initail_state, goal_state):
        self.INITIAL_STATE = initail_state
        self.GOAL_STATE = goal_state
    
    def PRINT(self, state):
        """Print the current state of the problem"""
        for i in range(len(state.STATE)):
            print(state.STATE[i]) # print every row
        print('\n')

    def GOAL_TEST(self, state):
        """Test if the goal state is reached"""
        return state == self.GOAL_STATE # if state is goal state, return True

class node:
    def __init__(self, state, parent=None):
        self.STATE = state
        self.MISPLACED_TILE = 0
        self.MANHATTAN_DISTANCE = 0

        if not parent:
            self.PARENT = None
            self.DEPTH = 0
        else:
            self.STATE = state
            self.PARENT = parent
            self.DEPTH = self.PARENT.DEPTH + 1
    
    def up(state, row, col):
        """Move blank tile up and return new state."""
        up_state = deepcopy(state)
        up_state[row][col], up_state[row - 1][col] = up_state[row - 1][col], up_state[row][col] # move blank tile up
        return up_state

    def down(state, row, col):
        """Move blank tile down and return new state."""
        down_state = deepcopy(state)
        down_state[row][col], down_state[row + 1][col] = down_state[row + 1][col], down_state[row][col] # move blank tile down
        return down_state

    def left(state, row, col):
        """Move blank tile left and return new state."""
        left_state = deepcopy(state)
        left_state[row][col], left_state[row][col - 1] = left_state[row][col - 1], left_state[row][col] # move blank tile left
        return left_state

    def right(state, row, col):
        """Move blank tile right and return new state."""
        right_state = deepcopy(state)
        right_state[row][col], right_state[row][col + 1] = right_state[row][col + 1], right_state[row][col] # move blank tile right
        return right_state
    
    # misplaced tile hueristic
    def misplaced(self):
        """Calculate and return the number of misplaced tiles in the current state."""
        if self.MISPLACED_TILE:
            return self.MISPLACED_TILE
        else:
            # create goal state to count misplaced tiles
            goal_list = [i for i in range(1, len(self.STATE)**2)] + [0]
            goal_state = [goal_list[i:i+n] for i in range(0, size + 1, n)]
            total = 0

            # calculate the total number of misplaced tiles in the state
            for i in range(len(self.STATE)):
                for j in range(len(self.STATE[i])):
                    if self.STATE[i][j] == 0: # don't consider blank tile
                        continue
                    if self.STATE[i][j] != goal_state[i][j]:
                        total += 1

            self.MISPLACED_TILE = total
            return self.MISPLACED_TILE
    
    # manhattan distance hueristic
    def manhattan(self):
        """Calculate and return the manhattan distance of the current state."""
        if self.MANHATTAN_DISTANCE:
            return self.MANHATTAN_DISTANCE
        else:
            total = 0
            for i in range(len(self.STATE)):
                for j in range(len(self.STATE[i])):
                    val = self.STATE[i][j]
                    if val != 0:
                        goal_i, goal_j = divmod(val - 1, len(self.STATE))  # calculate the goal position of the current value using quotient and remainder
                        distance = abs(i - goal_i) + abs(j - goal_j)  # calculate manhattan distance
                        total += distance
            
            self.MANHATTAN_DISTANCE = total
            return self.MANHATTAN_DISTANCE

    # tiebreaker in priority queue
    def __eq__(self, other):
        return self.DEPTH == other.DEPTH or self.DEPTH + self.MISPLACED_TILE == other.DEPTH + other.MISPLACED_TILE or self.DEPTH + self.MANHATTAN_DISTANCE == other.DEPTH + other.MANHATTAN_DISTANCE

def EXPAND(state):
    """
    Takes in a state, determines all possible operators and perform those actions.\n
    Return the list of expanded nodes
    """
    expanded_nodes = []
    operators = []
    row, col = next(((row, col.index(0)) for row, col in enumerate(state.STATE) if 0 in col), None) # get the blank position

    # check which operators are available for this state
    if row > 0:
        operators.append(node.up)
    if row < len(state.STATE) - 1:
        operators.append(node.down)
    if col > 0:
        operators.append(node.left)
    if col < len(state.STATE[0]) - 1:
        operators.append(node.right)

    # for each possible operator expand the node and add it to the list of expanded nodes
    for op in operators:
        new_node = op(state.STATE, row, col)
        # print(f'node state in expand {new_node}')
        expanded_nodes.append(node(new_node, state))
    
    # print(f'expanded nodes {expanded_nodes}')
    return expanded_nodes

def backtrack(goal):
    """
    Take in a goal state and backtrack upto the initial state.\n
    Return the solution path.\n
    """
    solution_path = [goal.STATE]
    curr = goal.PARENT
    while curr.PARENT:
        solution_path.append(curr.STATE)
        curr = curr.PARENT
    return solution_path

# main search function to be called for any algorithm
def general_search(problem, QUEUING_FUNCTION):
    """General search function which searches for a solution using any queuing function (uniform cost, misplaced tile, manhattan distance)"""
    nodes = [] # priority queue
    visited = {} # to check if a state is already visited
    nodes_expanded = 0 # how many nodes have been expanded
    max_queue_size = 0 # maximum size of the priority queue
    start_time = time.time() # start time of search
    depth_counts = {} # for plotting purposes

    heappush(nodes, (float('inf'), problem.INITIAL_STATE)) # add initial state to the queue with infinite cost

    while True:
        if not nodes:
            break

        max_queue_size = max(max_queue_size, nodes.__len__()) # keep track of maximum expanded nodes
        _, curr_node = heappop(nodes) # get the node with the lowest cost
        visited[tuple(tuple(row) for row in curr_node.STATE)] = True # mark node as visited

        # if goal state is reached, terminate and optianally return the solution path
        if problem.GOAL_TEST(curr_node.STATE):
            end_time = time.time() - start_time # end time of search
            print(f'Puzzle solved successfully!!')
            print(f'Answer found at {curr_node.DEPTH} depth. \n{nodes_expanded} nodes expanded. \n{max_queue_size} maximum queue size.')
            print(f'Search completed in {end_time:.2f} seconds')

            # print the solution path
            print(f'Enter 1 to print the solution or 0 to exit: ', end=' ')
            option = int(input())
            if option:
                solution_path = backtrack(curr_node)
                for i, state in enumerate(reversed(solution_path)): # print the solution path in reverse order (initial state --> goal state)
                    print(f'depth: {i+1}')
                    for i in range(len(state)):
                        print(state[i]) # print every row
            break
        # otherwise, expand the current node with all possible operators and add them to the priority queue
        else:
            for child_node in EXPAND(curr_node): # get the possible operators for this node
                check_node = tuple(tuple(row) for row in child_node.STATE)
                if check_node not in visited.keys(): # check if node is not visited
                    if QUEUING_FUNCTION == 'uniform_cost_search':
                        heappush(nodes, (child_node.DEPTH, child_node)) # cost is the current node's depth
                    if QUEUING_FUNCTION == 'misplaced_tile_search':
                        heappush(nodes, (child_node.DEPTH + child_node.misplaced(), child_node)) # cost is the current node's depth + misplaced tile
                    if QUEUING_FUNCTION == 'manhattan_distance_search':
                        heappush(nodes, (child_node.DEPTH + child_node.manhattan(), child_node)) # cost is the current node's depth + manhattan distance
                    nodes_expanded += 1 # keep track of how many nodes have been expanded

                    # for plotting purposes
                    depth = child_node.DEPTH
                    if depth in depth_counts:
                        depth_counts[depth] += 1
                    else:
                        depth_counts[depth] = 1
    
    return depth_counts

def get_puzzle_input():
    """Get the initial state of the puzzle from the user."""
    puzzle = []
    print(f'The row numbers should be space separated and between 1 and {size} (inclusive).\n')
    for i in range(n):
        print(f'Enter the {i+1}th row:', end=' ')
        row = list(map(int, input().strip().split())) # get a row
        puzzle.append(row) # and add it to the puzzle
    
    return size, n, puzzle

def get_goal_state(size, n):
    """Determine the goal state of the puzzle."""
    goal_list = [i for i in range(1, size + 1)] # create a goal list
    goal_list.append(0) # add blank tile
    goal_state = [goal_list[i:i+n] for i in range(0, size + 1, n)] # convert to the matrix representation
    print(f'goal state: {goal_state}')
    return goal_state

def is_solvable(puzzle):
    """Check if the given puzzle is solvable."""
    state = [i for row in puzzle for i in row if i != 0]
    inversions = 0
    sz = len(state)

    # bubble sort logic to count inversions
    for i in range(sz):
        for j in range(i+1, sz):
            if state[i] > state[j]:
                inversions += 1

    # if width is odd
    n = len(puzzle[0])
    if n % 2 == 1:
        if inversions % 2 == 0: # odd width should have even inversions
            return True
    # if width is even
    else:
        for i in range(n - 1, -1, -1):
            if 0 in puzzle[i]:
                idx = n - i # get the row index of the blank from bottom

        if idx % 2 == 1:
            if inversions % 2 == 0: # for even inversions, 0 (blank) should be at an odd row from bottom
                return True
        if idx % 2 == 0:
            if inversions % 2 == 1: # for odd inversions, 0 (blank) should be at an even row from bottom
                return True

    return False

def algorithm(puzzle):
    """
    Select the algorithm to solve the puzzle.\n
    1. Uniform Cost Search\n
    2. Misplaced Tile Search\n
    3. Manhattan Distance Search\n
    """
    # terminate if the puzzle is not solvable
    if not is_solvable(puzzle.INITIAL_STATE.STATE):
        raise AssertionError(f'The puzzle is not solvable. Try another input.')

    print(f'Enter 1 for Uniform Cost Search \nEnter 2 for A* with the Misplaced Tile heuristic \nEnter 3 for A* with the Manhattan Distance heuristic')
    print(f'Enter your choice: ', end=' ')
    algorithm = int(input())

    # match the input to the right algorithm
    match algorithm:
        case 1:
            general_search(puzzle, 'uniform_cost_search')
            # depth_counts = general_search(puzzle, 'uniform_cost_search')
        case 2:
            general_search(puzzle, 'misplaced_tile_search')
            # depth_counts = general_search(puzzle, 'misplaced_tile_search')
        case 3:
            general_search(puzzle, 'manhattan_distance_search')
            # depth_counts = general_search(puzzle, 'manhattan_distance_search')
    
    # return depth_counts # for plotting purposes

if __name__ == '__main__':
    print(f'Enter the puzzle size (8, 15, 24): ', end=' ')
    size = int(input()) # puzzle size e.g. 8, 15, 24
    n = int(math.sqrt(size)) + 1 # determine n to create a n*n grid

    # test inputs from project description
    trivial = [[1,2,3], [4,5,6], [7,8,0]]
    very_easy = [[1,2,3], [4,5,6], [7,0,8]]
    easy = [[1,2,0], [4,5,3], [7,8,6]]
    doable = [[0,1,2], [4,5,3], [7,8,6]]
    oh_boy = [[8,7,1], [6,0,2], [5,4,3]]
    unsolvable = [[1,2,3], [4,5,6], [8,7,0]]

    # get puzzle and run the algorithm chosen
    size, n, initial_state = get_puzzle_input()
    assert all(0 <= i <= size for row in initial_state for i in row), f'Invalid input for {size}-puzzle. Numbers should be between 0 and {size}.'
    assert len({i for row in initial_state for i in row}) == size + 1, f'Duplicate numbers found in input.'
    assert len(initial_state) == n and len(initial_state[0]) == n, f'Please enter a valid puzzle of size {n}.'

    goal_state = get_goal_state(size, n)
    puzzle = Problem(node(initial_state), goal_state)
    puzzle.PRINT(puzzle.INITIAL_STATE)
    algorithm(puzzle)
    # depth_counts = algorithm(puzzle) # for plotting purposes
    # print(f'depth counts {depth_counts}') # for plotting purposes
