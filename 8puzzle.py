import math
import time
import heapq

class Puzzle:
    def __init__(self, initail_state, goal_state):
        self.initial_state = initail_state
        self.goal_state = goal_state
    
    def print_state(self, state):
        for i in range(len(state)):
            print(state[i]) # print every row
    
    def is_goal_state(self, state):
        return state == self.goal_state

    def get_possible_operators(self, state):
        pass

    def next_state(self, state, operator):
        pass

def uniform_cost_search(puzzle):
    pass

def misplaced_tile_search(puzzle):
    pass

def manhattan_distance_search(puzzle):
    pass

# main search function to be called for any algorithm
def general_search(puzzle, queuing_function):
    nodes = []
    visited = set() # to check if a state is already visited
    depth = 0 # how deep we are in the search tree
    nodes_expanded = 0 # how many nodes have been expanded
    start_time = time.time() # start time of search

    heapq.heappush(nodes, puzzle.initial_state) # add initial state to the queue
    # print(nodes.pop())

    while nodes:
        node = nodes.pop()
        if puzzle.is_goal_state(node): # if node is the goal state
            end_time = time.time() - start_time # end time of search
            print(f'Puzzle solved successfully!!')
            print(f'Answer found at {depth} depth. \n{nodes_expanded} nodes expanded.')
            print(f'Search completed in {end_time:.2f} seconds')
            return node # return the goal state
        else:
            pass

def get_puzzle_input():
    puzzle = []
    for i in range(n):
        print(f'Enter the {i+1}th row:', end=' ')
        row = list(map(int, input().strip().split())) # get a row
        puzzle.append(row) # and add it to the puzzle
    
    return size, n, puzzle

def get_goal_state(size, n):
    goal_list = [i for i in range(1, size + 1)]
    goal_list.append(0)
    goal_state = [goal_list[i:i+n] for i in range(0, size + 1, n)]
    print(goal_state)
    return goal_state

# to check if a given state is solvable
# implemented from rules found in the following stackoverflow question
# https://stackoverflow.com/questions/55454496/is-it-possible-to-check-if-the-15-puzzle-is-solvable-with-a-different-goal-state
def is_solvable(puzzle):
    state = [i for row in puzzle for i in row if i != 0]
    inversions = 0
    sz = len(state)

    # bubble sort logic to count inversions
    for i in range(sz):
        for j in range(i+1, sz):
            if state[i] > state[j]:
                inversions += 1
    print(f'inversions: {inversions}')

    # if width is odd
    n = len(puzzle[0])
    if n % 2 == 1:
        if inversions % 2 == 0: # odd width should have even inversions
            return True
    # if width is even
    else:
        for i in range(n - 1, -1, -1):
            if 0 in puzzle[i]:
                idx = (n - i - 1) % 2 # for even width get the row index of the blank
                print(f'row index of 0: {idx}')

        if idx % 2 == 1:
            if inversions % 2 == 0: # for even inversions, 0 should be at an odd row from bottom
                return True
        if idx % 2 == 0:
            if inversions % 2 == 1: # for odd inversions, 0 should be at an even row from bottom
                return True

    return False

def algorithm(puzzle):
    # terminate if the puzzle is not solvable
    if not is_solvable(puzzle):
        raise AssertionError(f'The puzzle is not solvable. Try another input.')

    print(f'Enter 1 for Uniform Cost Search \n Enter 2 for A* with the Misplaced Tile heuristic \n Enter 3 for A* with the Manhattan Distance heuristic')
    print(f'Enter your choice: ', end=' ')
    algorithm = int(input())

    # match the input to the right algorithm
    match algorithm:
        case 1:
            general_search(puzzle, uniform_cost_search)
        case 2:
            general_search(puzzle, misplaced_tile_search)
        case 3:
            general_search(puzzle, manhattan_distance_search)

if __name__ == '__main__':
    print(f'Enter the puzzle size: ', end=' ')
    size = int(input()) # puzzle size e.g. 8, 15, 24
    n = int(math.sqrt(size)) + 1 # determine n to create a n*n grid

    # test inputs
    trivial = [[1,2,3], [4,5,6], [7,8,0]]
    very_easy = [[1,2,3], [4,5,6], [7,0,8]]
    easy = [[1,2,0], [4,5,3], [7,8,6]]
    doable = [[0,1,2], [4,5,3], [7,8,6]]
    oh_boy = [[8,7,1], [6,0,2], [5,4,3]]
    final_state = [[1,2,3], [4,5,6], [7,8,0]]

    # get puzzle and run the algorithm chosen
    size, n, initial_state = get_puzzle_input()
    assert all(0 <= i <= size for row in initial_state for i in row), f'Invalid input for 8-puzzle. Numbers should be between 0 and {size}.'
    assert len({i for row in initial_state for i in row}) == size + 1, f'Duplicate numbers found in input.'
    assert len(initial_state) == n and len(initial_state[0]) == n, f'Please enter a valid puzzle of size {n}.'

    # algorithm(puzzle)
    goal_state = get_goal_state(size, n)
    puzzle = Puzzle(initial_state, goal_state)
    puzzle.print_state(initial_state)
    general_search(puzzle, uniform_cost_search)
