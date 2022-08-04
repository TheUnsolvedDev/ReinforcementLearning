import gym
import numpy as np
import time

from params import *


def code_to_mark(code):
    return CODE_MARK_MAP[code]


def mark_to_code(mark):
    return BLACK_DISK if mark == 'B' else WHITE_DISK


def next_mark(mark):
    return 'W' if mark == 'B' else 'B'


def agent_by_mark(agents, mark):
    for agent in agents:
        if agent.mark == mark:
            return agent


def next_state_show(state, action):
    board, mark = state
    nboard = list(board[:])
    nboard[action] = mark_to_code(mark)
    nboard = tuple(nboard)
    return nboard, next_mark(mark)


class Reversi_env(gym.Env):
    metadata = {"render.modes": ["np_array", "human"]}

    def __init__(self, size=6):
        self.size = size
        self.action_space = gym.spaces.Discrete(self.size ** 2 + 2)
        self.board = self.create_board()
        self.observation_space = gym.spaces.Box(
            np.zeros([self.size] * 2), np.ones([self.size] * 2))
        self.termination = False
        self.possible_actions_in_obs = False
        self.sudden_death_on_invalid_move = True
        self.mute = False
        self.viewer = None
        self.num_disk_as_reward = True

        # Initialize internal states.
        self.mark = 'W'
        self.player_turn = WHITE_DISK
        self.winner = NO_DISK
        self.terminated = False
        self.possible_moves = []

    def get_observation(self):
        if self.player_turn == WHITE_DISK:
            # White turn, we don't negate state since white=1.
            state = self.board
        else:
            # Black turn, we negate board state such that black=1.
            state = -self.board
        if self.possible_actions_in_obs:
            grid_of_possible_moves = np.zeros(self.size ** 2, dtype=bool)
            grid_of_possible_moves[self.possible_moves] = True
            return tuple(np.concatenate([np.expand_dims(state, axis=0),
                                         grid_of_possible_moves.reshape(
                [1, self.size, self.size])],
                axis=0).flatten()), self.mark
        else:
            return tuple(state.flatten()), self.mark

    def create_board(self):
        board = np.zeros([self.size] * 2, dtype=int)
        board[self.size//2 - 1][self.size//2 - 1] = WHITE_DISK
        board[self.size//2][self.size//2] = WHITE_DISK
        board[self.size//2 - 1][self.size//2] = BLACK_DISK
        board[self.size//2][self.size//2 - 1] = BLACK_DISK
        return board

    def reset(self):
        self.board = self.create_board()
        self.player_turn = BLACK_DISK
        self.winner = NO_DISK
        self.terminated = False
        self.possible_moves = self.get_possible_actions()
        return self.get_observation()

    def get_possible_actions(self, board=None):
        actions = []
        if board is None:
            if self.player_turn == WHITE_DISK:
                board = self.board
            else:
                board = -self.board

        for row_ix in range(self.size):
            for col_ix in range(self.size):
                if board[row_ix][col_ix] == NO_DISK:
                    if (
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 1, 1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 1, 0) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 1, -1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 0, 1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 0, -1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, -1, 1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, -1, 0) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, -1, -1)
                    ):
                        actions.append(row_ix * self.size + col_ix)
        return actions

    def get_num_killed_enemy(self, board, x, y, delta_x, delta_y):
        # We overload WHITE_DISK to be our disk, and BLACK_DISK to be enemies.
        # (x, y) is a valid position if the following pattern exists:
        #    "(x, y), BLACK_DISK, ..., BLACK_DISK, WHITE_DISK"

        next_x = x + delta_x
        next_y = y + delta_y

        # The neighbor must be an enemy.
        if (
                next_x < 0 or
                next_x >= self.size or
                next_y < 0 or
                next_y >= self.size or
                board[next_x][next_y] != BLACK_DISK
        ):
            return 0

        # Keep scanning in the direction.
        cnt = 0
        while (
                0 <= next_x < self.size and
                0 <= next_y < self.size and
                board[next_x][next_y] == BLACK_DISK
        ):
            next_x += delta_x
            next_y += delta_y
            cnt += 1

        if (
                next_x < 0 or
                next_x >= self.size or
                next_y < 0 or
                next_y >= self.size or
                board[next_x][next_y] != WHITE_DISK
        ):
            return 0
        else:
            return cnt

    def get_possible_actions(self, board=None):
        actions = []
        if board is None:
            if self.player_turn == WHITE_DISK:
                board = self.board
            else:
                board = -self.board

        for row_ix in range(self.size):
            for col_ix in range(self.size):
                if board[row_ix][col_ix] == NO_DISK:
                    if (
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 1, 1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 1, 0) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 1, -1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 0, 1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 0, -1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, -1, 1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, -1, 0) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, -1, -1)
                    ):
                        actions.append(row_ix * self.size + col_ix)
        return actions

    def print_board(self, print_valid_moves=True):
        valid_actions = self.get_possible_actions()

        if print_valid_moves:
            board = self.board.copy().ravel()
            for p in valid_actions:
                board[p] = 2
            board = board.reshape(*self.board.shape)
        else:
            board = self.board

        print('Turn: {}'.format(
            'WHITE' if self.player_turn == WHITE_DISK else 'BLACK'))
        print('Valid actions: {}'.format(valid_actions))
        for row in board:
            print(' '.join(map(lambda x: ['B', '+', 'W', 'V'][x + 1], row)))
        print('-' * 10)

    def set_board_state(self, board, perspective=WHITE_DISK):
        """Force setting the board state, necessary in model-based RL."""
        if np.ndim(board) > 2:
            state = board[0]
        else:
            state = board
        if perspective == WHITE_DISK:
            self.board = np.array(state)
        else:
            self.board = -np.array(state)

    def update_board(self, action):
        x = action // self.size
        y = action % self.size

        if self.player_turn == BLACK_DISK:
            self.board = -self.board

        for delta_x in [-1, 0, 1]:
            for delta_y in [-1, 0, 1]:
                if not (delta_x == 0 and delta_y == 0):
                    kill_cnt = self.get_num_killed_enemy(
                        self.board, x, y, delta_x, delta_y)
                    for i in range(kill_cnt):
                        dx = (i + 1) * delta_x
                        dy = (i + 1) * delta_y
                        self.board[x + dx][y + dy] = WHITE_DISK
        self.board[x][y] = WHITE_DISK

        if self.player_turn == BLACK_DISK:
            self.board = -self.board

    def step(self, action):

        # Apply action.
        if self.terminated:
            raise ValueError('Game has terminated!')
        if action not in self.possible_moves:
            invalid_action = True
        else:
            invalid_action = False
        if not invalid_action:
            self.update_board(action)

        # Determine if game should terminate.
        num_vacant_positions = (self.board == NO_DISK).sum()
        no_more_vacant_places = num_vacant_positions == 0
        sudden_death = invalid_action and self.sudden_death_on_invalid_move
        done = sudden_death or no_more_vacant_places

        current_player = self.player_turn
        if done:
            # If game has terminated, determine winner.
            self.winner = self.determine_winner(sudden_death=sudden_death)
        else:
            # If game continues, determine who moves next.
            self.set_player_turn(-self.player_turn)
            if len(self.possible_moves) == 0:
                self.set_player_turn(-self.player_turn)
                if len(self.possible_moves) == 0:
                    if not self.mute:
                        print('No possible moves for either party.')
                    self.winner = self.determine_winner()

        reward = 0
        if self.terminated:
            if self.num_disk_as_reward:
                if sudden_death:
                    # Strongly discourage invalid actions.
                    reward = -(self.size ** 2)
                else:
                    white_cnt, black_cnt = self.count_disks()
                    if current_player == WHITE_DISK:
                        reward = white_cnt - black_cnt
                        if black_cnt == 0:
                            reward = self.size ** 2
                    else:
                        reward = black_cnt - white_cnt
                        if white_cnt == 0:
                            reward = self.size ** 2
            else:
                reward = self.winner * current_player
        self.mark = next_mark(self.mark)
        return self.get_observation(), reward, self.terminated, None

    def set_player_turn(self, turn):
        self.player_turn = turn
        self.possible_moves = self.get_possible_actions()

    def count_disks(self):
        white_cnt = (self.board == WHITE_DISK).sum()
        black_cnt = (self.board == BLACK_DISK).sum()
        return white_cnt, black_cnt

    def determine_winner(self, sudden_death=False):
        self.terminated = True
        if sudden_death:
            if not self.mute:
                print('sudden death due to rule violation')
            if self.player_turn == WHITE_DISK:
                if not self.mute:
                    print('BLACK wins')
                return BLACK_DISK
            else:
                if not self.mute:
                    print('WHITE wins')
                return WHITE_DISK
        else:
            white_cnt, black_cnt = self.count_disks()
            if not self.mute:
                print('white: {}, black: {}'.format(white_cnt, black_cnt))
            if white_cnt > black_cnt:
                if not self.mute:
                    print('WHITE wins')
                return WHITE_DISK
            elif black_cnt > white_cnt:
                if not self.mute:
                    print('BLACK wins')
                return BLACK_DISK
            else:
                if not self.mute:
                    print('DRAW')
                return NO_DISK

    def render(self, mode='np_array', close=False):
        if close:
            return
        if mode == 'np_array':
            self.print_board()
        else:
            self.show_gui_board()
            # pass


if __name__ == '__main__':
    obj = Reversi_env(6)
    state = obj.reset()
    done = False

    while not done:
        action = np.random.choice(obj.get_possible_actions())
        state, reward, done, info = obj.step(action)
        print(state, reward, done, info)
        obj.render()
        time.sleep(0.2)
