
import gym
from gym import spaces
import numpy as np
import sys
import pygame
import time
import itertools
import tqdm
import pickle

# Create the constants (go ahead and experiment with different values)
BOARDWIDTH = 3  # number of columns in the board
BOARDHEIGHT = 3  # number of rows in the board
TILESIZE = 80
WINDOWWIDTH = 640
WINDOWHEIGHT = 480
FPS = 30
BLANK = None

#                 R    G    B
BLACK = (0,   0,   0)
WHITE = (255, 255, 255)
BRIGHTBLUE = (0,  50, 255)
DARKTURQUOISE = (3,  54,  73)
GREEN = (0, 204,   0)

BGCOLOR = DARKTURQUOISE
TILECOLOR = GREEN
TEXTCOLOR = WHITE
BORDERCOLOR = BRIGHTBLUE
BASICFONTSIZE = 20

BUTTONCOLOR = WHITE
BUTTONTEXTCOLOR = BLACK
MESSAGECOLOR = WHITE

XMARGIN = int((WINDOWWIDTH - (TILESIZE * BOARDWIDTH + (BOARDWIDTH - 1))) / 2)
YMARGIN = int(
    (WINDOWHEIGHT - (TILESIZE * BOARDHEIGHT + (BOARDHEIGHT - 1))) / 2)

solve_thresh = 1000
# 0 (swap up),
# 1 (swap right),
# 2 (swap left),
# 3 (swap down)


class NPuzzle(gym.Env):
    metadata = {'render.modes': ['human', 'asci'],"render_fps": 30,}

    def __init__(self, n=3):
        super(NPuzzle, self).__init__()

        self.n = n
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 9, (9, ))
        self.state = self.shuflled_state()
        self.last_action = None
        self.last_reward = None
        self.steps_beyond_done = None
        self.reset()
        self.viewer = None

        with open('transition.pkl', 'rb') as transition:
            self.P = pickle.load(transition)

        with open('id_to_state.pkl', 'rb') as id_to_state:
            self.its = pickle.load(id_to_state)

        with open('state_to_id.pkl', 'rb') as state_to_id:
            self.sti = pickle.load(state_to_id)
             
        self.window_surface = None
        self.clock = None

    def shuflled_state(self):
        tiles = list(range(self.n**2))
        np.random.shuffle(tiles)

        while not self.is_solvable(tiles):
            np.random.shuffle(tiles)

        return np.array(tiles).reshape((self.n, self.n))

    def is_solvable(self, puzzle):
        inversions = 0

        for i in range(len(puzzle)):
            for j in range(i + 1, len(puzzle)):
                if puzzle[i] > puzzle[j] and puzzle[i] != 0 and puzzle[j] != 0:
                    inversions += 1

        return inversions % 2 == 0

    def step(self, action):
        self.steps_beyond_done += 1
        assert self.action_space.contains(
            action), "%r (%s) invalid " % (action, type(action))

        blank = np.where(self.state == 0)
        b_x = blank[0][0]
        b_y = blank[1][0]

        # Move Up
        if action == 0 and b_x > 0:
            self.state[b_x][b_y] = self.state[b_x - 1][b_y]
            self.state[b_x - 1][b_y] = 0

        # Move Right
        elif action == 1 and b_y < self.n-1:
            self.state[b_x][b_y] = self.state[b_x][b_y + 1]
            self.state[b_x][b_y + 1] = 0

        # Move Down
        elif action == 2 and b_x < self.n-1:
            self.state[b_x][b_y] = self.state[b_x + 1][b_y]
            self.state[b_x + 1][b_y] = 0

        # Move Left
        elif action == 3 and b_y > 0:
            self.state[b_x][b_y] = self.state[b_x][b_y - 1]
            self.state[b_x][b_y - 1] = 0

        # Determine reward (negative Manhattan distance to goal state)
        distance = 0

        for i, row in enumerate(self.state):
            for j, tile in enumerate(row):
                correct_row = np.floor(tile / self.n) - \
                    (1 if tile % self.n == 0 else 0)
                correct_col = (tile % self.n - 1) % self.n

                if tile == 0:
                    correct_row = self.n - 1
                    correct_col = self.n - 1

                distance += abs(i - correct_row) + abs(j - correct_col)

        done = distance == 0
        if distance > 0:
            distance = -distance/10
        else:
            distance = 20
        
        reward = distance#0 if distance else 1

        self.last_action = action
        self.last_reward = reward

        if done:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                return self.state.flatten(), reward, done, {}

        if self.steps_beyond_done == solve_thresh:
            done = True
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                return self.state.flatten(), reward, done, {}

        return self.state.flatten(), reward, done, {}

    def reset(self):
        self.state = self.shuflled_state()
        self.steps_beyond_done = 0
        return self.state.flatten()

    def print_board(self):
        for i in range(self.n):
            print('|', end='')
            for j in range(self.n):
                if self.state[i][j] == 0:
                    print('  ', end='|')
                if len(str(self.state[i][j])) == 2:
                    print(''+str(self.state[i][j]), end='|')
                if len(str(self.state[i][j])) == 1 and self.state[i][j] != 0:
                    print(' '+str(self.state[i][j]), end='|')

                if j == self.n-1:
                    print()
        print()
        
    def render_human(self):
        if self.window_surface is None:
            pygame.init()
            pygame.display.init()

            pygame.display.set_caption('Slide Puzzle')
            self.BASICFONT = pygame.font.Font('freesansbold.ttf', BASICFONTSIZE)

            self.window_surface = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        def getLeftTopOfTile(tileX, tileY):
            left = XMARGIN + (tileX * TILESIZE) + (tileX - 1)
            top = YMARGIN + (tileY * TILESIZE) + (tileY - 1)
            return (left, top)

        def drawTile(tilex, tiley, number, adjx=0, adjy=0):
            # draw a tile at board coordinates tilex and tiley, optionally a few
            # pixels over (determined by adjx and adjy)
            left, top = getLeftTopOfTile(tilex, tiley)
            pygame.draw.rect(self.window_surface, TILECOLOR,
                             (left + adjx, top + adjy, TILESIZE, TILESIZE))
            textSurf = self.BASICFONT.render(str(number), True, TEXTCOLOR)
            textRect = textSurf.get_rect()
            textRect.center = left + \
                int(TILESIZE / 2) + adjx, top + int(TILESIZE / 2) + adjy
            self.window_surface.blit(textSurf, textRect)

        def makeText(text, color, bgcolor, top, left):
            # create the Surface and Rect objects for some text.
            textSurf = self.BASICFONT.render(text, True, color, bgcolor)
            textRect = textSurf.get_rect()
            textRect.topleft = (top, left)
            return (textSurf, textRect)

        message = 'Have Fun'
        self.window_surface.fill(BGCOLOR)
        textSurf, textRect = makeText(message, MESSAGECOLOR, BGCOLOR, 5, 5)
        self.window_surface.blit(textSurf, textRect)
        board = self.state.T
        for tilex in range(len(board)):
            for tiley in range(len(board[0])):
                if board[tilex][tiley]:
                    drawTile(tilex, tiley, board[tilex][tiley])

        left, top = getLeftTopOfTile(0, 0)
        width = BOARDWIDTH * TILESIZE
        height = BOARDHEIGHT * TILESIZE
        pygame.draw.rect(self.window_surface, BORDERCOLOR, (left - 5,
                         top - 5, width + 11, height + 11), 4)

        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def render(self, mode='asci', close=False):
        if close:
            return

        if mode == 'asci':
            # print("\n".join('|'.join(map(str, row)).replace('0', ' ')
            #             for row in self.state) + "\n")
            self.print_board()
            if self.last_action is not None:
                print("{0} {1}\n".format(self.last_reward, [
                    "U", "R", "D", "L"][self.last_action]))
            else:
                print("\n")

        if mode == 'human':
            self.render_human()


if __name__ == "__main__":
    env = NPuzzle(3)
    env.reset()

    done = False
    while not done:
        time.sleep(0.2)
        env.render(mode='asci')
        action = np.random.randint(0, 4)
        next_state, reward, done, info = env.step(action)

    env.close()
