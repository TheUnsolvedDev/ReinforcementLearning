from dis import dis
import tqdm
import numpy as np
import itertools
import pickle


def is_solvable(puzzle):
    inversions = 0

    for i in range(len(puzzle)):
        for j in range(i + 1, len(puzzle)):
            if puzzle[i] > puzzle[j] and puzzle[i] != 0 and puzzle[j] != 0:
                inversions += 1

    return inversions % 2 == 0


def reward(puzzle, n=3):
    distance = 0
    puzzle = np.array(puzzle).reshape((n, n))

    for i, row in enumerate(puzzle):
        for j, tile in enumerate(row):
            correct_row = np.floor(tile / n) - \
                (1 if tile % n == 0 else 0)
            correct_col = (tile % n - 1) % n

            if tile == 0:
                correct_row = n - 1
                correct_col = n - 1

            distance += abs(i - correct_row) + abs(j - correct_col)
            
    if distance > 0:
        distance = -distance/10
    else:
        distance = 20
    return distance


def step(puzzle, action, n=3):
    puzzle = np.array(puzzle).reshape((n, n))
    blank = np.where(puzzle == 0)
    b_x = blank[0][0]
    b_y = blank[1][0]

    if action == 0 and b_x > 0:
        puzzle[b_x][b_y] = puzzle[b_x - 1][b_y]
        puzzle[b_x - 1][b_y] = 0

    # Move Right
    elif action == 1 and b_y < n-1:
        puzzle[b_x][b_y] = puzzle[b_x][b_y + 1]
        puzzle[b_x][b_y + 1] = 0

    # Move Down
    elif action == 2 and b_x < n-1:
        puzzle[b_x][b_y] = puzzle[b_x + 1][b_y]
        puzzle[b_x + 1][b_y] = 0

    # Move Left
    elif action == 3 and b_y > 0:
        puzzle[b_x][b_y] = puzzle[b_x][b_y - 1]
        puzzle[b_x][b_y - 1] = 0

    return puzzle.flatten()


def calculate_P():
    permuts = list(itertools.permutations(list(range(9))))

    solvables = []
    rewards = []
    for elem in permuts:
        if is_solvable(elem):
            solvables.append(elem)
            rewards.append(reward(elem))

    P = {}
    for puzzle in tqdm.tqdm(solvables):
        P[puzzle] = {}
        for action in range(4):
            prob = 1
            next_state = step(puzzle, action)
            rew = reward(puzzle)
            if rew == 0:
                done = True
            else:
                done = False
            P[puzzle][action] = (prob, next_state, rew, done)
    return P


def id_state():
    permuts = list(itertools.permutations(list(range(9))))

    solvables = []
    for elem in permuts:
        if is_solvable(elem):
            solvables.append(elem)

    def state_to_id():
        sti = {}
        for ind,elem in enumerate(solvables):
            sti[elem] = ind
        return sti

    def id_to_state():
        its = {}
        for ind,elem in enumerate(solvables):
            its[ind] = elem
        return its
    return state_to_id(),id_to_state()
    
if __name__ == '__main__':
    p = calculate_P()
    with open('transition.pkl','wb') as f:
        pickle.dump(p,f)
        
    sti,its = id_state()
    with open('state_to_id.pkl','wb') as f:
        pickle.dump(sti,f)
        
    with open('id_to_state.pkl','wb') as f:
        pickle.dump(its,f)
    
    # print(p)
    print('done')