# encoding: utf-8
from __future__ import print_function
import numpy
import random
# try:
#     from colors import red, blue
# except Exception as e:
#     print("Install 'colors' module for color",e)
def red(s): return s
def blue(s): return s


class Game:
    ROWS = 6
    COLS = 7

    def __init__(self):
        self.board = numpy.zeros((self.COLS, self.ROWS), dtype='i')
        self.heights = numpy.zeros(self.COLS, dtype='i')
        self.turn = 1
        self.history = []

    # is x, y on the board
    def on_board(self, x, y):
        return x >= 0 and x < self.COLS and y >= 0 and y < self.ROWS

    # count the number of pieces that match the piece on (x, y)
    # in direction (dx, dy)
    def scan(self, x, y, dx, dy):
        c = 0
        p = self.board[x, y]
        while self.on_board(x, y) and self.board[x, y] == p:
            c += 1
            x += dx
            y += dy
        return c

    # check whether or not the game is over
    def check_win(self, x):
        y = self.heights[x] - 1
        if self.scan(x, y, 0, -1) >= 4:
            return True
        if self.scan(x, y, 1, 1) + self.scan(x, y, -1, -1) - 1 >= 4:
            return True
        if self.scan(x, y, 1, -1) + self.scan(x, y, -1, 1) - 1 >= 4:
            return True
        if self.scan(x, y, 1, 0) + self.scan(x, y, -1, 0) - 1 >= 4:
            return True
        return False

    # make a move in column x
    def make_move(self, x):
        self.board[x, self.heights[x]] = self.turn
        self.heights[x] += 1
        self.turn *= -1
        self.history.append(x)
        return self.check_win(x)

    # unmake the last move in column x
    def unmake_move(self):
        x = self.history.pop()
        self.heights[x] -= 1
        self.board[x, self.heights[x]] = 0
        self.turn *= -1

    # return a list of available moves
    def moves(self):
        return [x for x in range(self.COLS) if self.heights[x] < self.ROWS]

    # print the board
    def show(self):
        print("Player one: ●           Player two: △")
        for y in range(self.ROWS):
            print('|', end='')
            for x in range(self.COLS):
                if self.board[x, self.ROWS - y - 1] == 1:
                    print(red('●') + '|', end='')
                elif self.board[x, self.ROWS - y - 1] == -1:
                    print(blue('△') + '|', end='')
                else:
                    print(' |', end='')
            print('')
        print('+-+-+-+-+-+-+-+')
        if len(self.history) > 0:
            print(' ', end='')
            last_move = self.history[-1]
            for x in range(self.COLS):
                if last_move == x:
                    print('^', end='')
                else:
                    print('  ', end='')
            print('')

if __name__ == '__main__':
    pass