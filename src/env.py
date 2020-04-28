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

DONE_REWARD = 0
GOAL_REWARD = 100.0
VALID_STEP_REWARD = -1.0
INVALID_STEP_REWARD = -5.0

MAX_STEPS = 100

MAP = [
    "GSGGRGGRGRGGRGGG",  # TODO: change S to R upon random start
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


def generate_random_map():
    pass


class DeliveryRouteEnv(Env):

    """
    Represents local neighborhood for package delivery.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):

        desc = np.asarray(MAP, dtype='c')
        nrow, ncol = desc.shape

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        p = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return row, col

        # Create transition matrix
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = p[s][a]
                    letter = desc[row, col]
                    if letter in b'F':
                        li.append((1.0, s, DONE_REWARD, True))
                    else:
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]

                        # Invalid move
                        if bytes(newletter) in b'G':
                            li.append((1.0, s, INVALID_STEP_REWARD, False))
                        else:
                            done = bytes(newletter) in b'G'
                            rew = GOAL_REWARD if newletter == b'G' else VALID_STEP_REWARD
                            li.append((1.0, newstate, rew, done))

        self.desc = desc
        self.nrow, self.ncol = nrow, ncol

        self.p = p
        self.isd = isd
        self.lastaction = None  # for rendering
        self.nS = nS
        self.nA = nA

        self.remaining_steps = MAX_STEPS

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        self.remaining_steps = MAX_STEPS
        return self.s

    def step(self, a):
        transitions = self.p[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
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
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        time.sleep(wait)

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
