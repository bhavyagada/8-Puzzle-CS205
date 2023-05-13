import math

def print_state(puzzle):
    for i in range(len(puzzle)):
        print(puzzle[i]) # print every row

class Problem:
    def __init__(self, INITIAL_STATE, GOAL_STATE):
        self.INITIAL_STATE = INITIAL_STATE
        self.GOAL_STATE = GOAL_STATE
    
    def GOAL_TEST(self, STATE):
        return STATE == self.GOAL_STATE

    def OPERATORS(self, STATE):
        pass

    def NEXT_STATE(self, STATE, OPERATOR):
        pass

def uniform_cost_search(puzzle):
    pass

def misplaced_tile_search(puzzle):
    pass

def manhattan_distance_search(puzzle):
    pass

# main search function to be called for any algorithm
def general_search(puzzle, queuing_function):
    pass

def get_puzzle_input():
    print(f'Enter the puzzle size: ', end=' ')
    size = int(input()) # puzzle size e.g. 8, 15, 24
    n = int(math.sqrt(size)) + 1 # determine n to create a n*n grid
    puzzle = []
    for i in range(n):
        print(f'Enter the {i+1}th row:', end=' ')
        row = list(map(int, input().strip().split())) # get a row
        puzzle.append(row) # and add it to the puzzle
    
    return size, n, puzzle

def algorithm(puzzle):
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
    # test inputs
    trivial = [[1,2,3], [4,5,6], [7,8,0]]
    very_easy = [[1,2,3], [4,5,6], [7,0,8]]
    easy = [[1,2,0], [4,5,3], [7,8,6]]
    doable = [[0,1,2], [4,5,3], [7,8,6]]
    oh_boy = [[8,7,1], [6,0,2], [5,4,3]]
    final_state = [[1,2,3], [4,5,6], [7,8,0]]

    # get puzzle and run the algorithm chosen
    size, n, puzzle = get_puzzle_input()
    assert all(0 <= i <= size for row in puzzle for i in row), f'Invalid input for 8-puzzle. Numbers should be between 0 and {size}.'
    assert len({i for row in puzzle for i in row}) == size + 1, f'Duplicate numbers found in input.'
    assert len(puzzle) == n and len(puzzle[0]) == n, f'Please enter a valid puzzle of size {n}.'
    print_state(puzzle)
    algorithm(puzzle)
