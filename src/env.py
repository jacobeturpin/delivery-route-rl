"""Delivery Route Environment"""

import sys
import time
from contextlib import closing
from io import StringIO

import numpy as np
from gym import Env, spaces, utils
from gym.envs.toy_text.discrete import categorical_sample
from gym.utils import seeding

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAX_STEPS = 100

DONE_REWARD = 0
VALID_STEP_REWARD = -1.0
INVALID_STEP_REWARD = -5.0
OUT_OF_BOUNDS_REWARD = -2.0
GOAL_REWARD = MAX_STEPS * 1.0

MAP = [
    "GRGGRGGRGRGGRGGG",
    "RRRRRRRRRRRRRRRR",
    "GRGGRGGGGGGGGGRG",
    "GRGGRGGGGGGGGGRG",

    "RRRRRRRRRRRRRRRR",
    "GRGGRGGRGGGGGGRR",
    "GRGGRGGRGGGGGGRG",
    "RRRRRRRRRRRRGGRG",

    "GRGGRGGRGGRGGGRR",
    "GRGGRGGRGGRGGGRG",
    "RRRRRRRRGGRGGGRG",
    "GRGGRGGGGGGGRRRR",

    "GRGGRGGGGGGGRGRG",
    "RRRRRGGGGGGGRGRG",
    "GRGGRRRRRRRRRGFR",
    "GRGGRGGGGGGGRGGG"
]


def generate_random_start(base_map):
    while True:
        nrow, ncol = base_map.shape
        r, c = np.random.choice(nrow), np.random.choice(ncol)
        if bytes(base_map[r, c]) in b'R':
            base_map[r, c] = b'S'
            return base_map


class DeliveryRouteEnv(Env):

    """
    Represents local neighborhood for package delivery.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, env_map, random_start=False):

        self.base_map = env_map
        self.random_start = random_start
        self.desc = self._set_start()
        self.nrow, self.ncol = self.desc.shape

        self.nA = 4
        self.nS = self.nrow * self.ncol

        self.p, self.isd = self._create_transition_matrix()
        self.last_action = None  # for rendering

        self.remaining_steps = MAX_STEPS

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)

    def _to_s(self, row, col):
        return row * self.ncol + col

    def _inc(self, row, col, a):
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif a == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif a == UP:
            row = max(row - 1, 0)
        return row, col

    def _set_start(self):
        char_map = np.asarray(self.base_map, dtype='c')

        if self.random_start:
            char_map = generate_random_start(char_map)
        else:
            char_map[0, 1] = b'S'
        return char_map

    def _create_transition_matrix(self):
        isd = np.array(self.desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        p = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        # Create transition matrix
        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self._to_s(row, col)
                for a in range(4):
                    li = p[s][a]
                    letter = self.desc[row, col]
                    if letter in b'F':
                        li.append((1.0, s, DONE_REWARD, True))
                    else:
                        newrow, newcol = self._inc(row, col, a)
                        newstate = self._to_s(newrow, newcol)
                        newletter = self.desc[newrow, newcol]

                        # Moving outside boundary
                        if s == newstate and bytes(newletter) in b'SR':
                            li.append((1.0, s, OUT_OF_BOUNDS_REWARD, False))

                        # Invalid move
                        elif bytes(newletter) in b'G':
                            li.append((1.0, s, INVALID_STEP_REWARD, False))

                        # Valid move (either done or not done)
                        else:
                            done = bytes(newletter) in b'F'
                            rew = GOAL_REWARD if done else VALID_STEP_REWARD
                            li.append((1.0, newstate, rew, done))
        return p, isd

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.desc = self._set_start()
        self.p, self.isd = self._create_transition_matrix()

        self.s = categorical_sample(self.isd, self.np_random)
        self.last_action = None
        self.remaining_steps = MAX_STEPS
        return self.s

    def step(self, a):
        transitions = self.p[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.last_action = a
        self.remaining_steps -= 1

        if self.remaining_steps <= 0:
            d = True

        return s, r, d, {"prob": p}

    def render(self, mode='human', wait=0):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.last_action is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.last_action]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        time.sleep(wait)

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
