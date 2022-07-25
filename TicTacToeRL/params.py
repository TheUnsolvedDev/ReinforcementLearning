import logging

MAX_EPISODES = 200000
MODEL_FILE = 'best_td_agent.dat'
DECAY = 1/MAX_EPISODES
DEFAULT_VALUE = 0
EPSILON = 0.8
GAMMA = 0.9999
ALPHA = 0.02
CODE_MARK_MAP = {0: ' ', 1: 'O', 2: 'X'}
NUM_LOC = 9
O_REWARD = 1
X_REWARD = -1
NO_REWARD = 0

LEFT_PAD = '  '
LOG_FMT = logging.Formatter('%(levelname)s '
                            '[%(filename)s:%(lineno)d] %(message)s',
                            '%Y-%m-%d %H:%M:%S')
