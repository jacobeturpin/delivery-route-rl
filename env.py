"""Delivery Route Environment"""

from gym import Env

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class DeliveryRouteEnv(Env):

    """
    Represents local neighborhood for package delivery.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        pass
