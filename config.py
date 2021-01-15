import collections
from typing import Dict, List, Optional, NamedTuple

import torch

from environment import Game

MAXIMUM_FLOAT_VALUE = float('inf')
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])
class MuZeroConfig(object):

    def __init__(self,
               action_space_size: int,
               max_moves: int,  #max game moves
               discount: float,
               dirichlet_alpha: float,
               num_simulations: int,  #mcts moves
               batch_size: int,
               td_steps: int,
               lr_init: float,
               lr_decay_steps: float,
               visit_softmax_temperature_fn,
               known_bounds: Optional[KnownBounds] = None,
               device = None):
        ### Self-Play
        self.action_space_size = action_space_size
        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.training_steps = 1000000
        self.checkpoint_interval = 1000
        self.window_size = 100   #replay buffer size
        self.batch_size = batch_size
        self.num_unroll_steps = 4
        self.td_steps = td_steps

        self.weight_decay = 1e-4

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_steps = lr_decay_steps

        self.device = device

    def new_game(self):
        return Game(self.action_space_size, self.discount)


def make_board_game_config(max_moves=9) -> MuZeroConfig:

    def visit_softmax_temperature(num_moves):
        if num_moves < min(2*max_moves//3, 30):
            return 1.0
        else:
            return 0.0  # Play according to the max.

    return MuZeroConfig(
        action_space_size=9,                                       
        max_moves=max_moves,                                               
        discount=1.0,
        dirichlet_alpha=0.1,
        num_simulations=50,
        batch_size=32,
        td_steps=max_moves,  # Always use Monte Carlo return for board games.              
        lr_init=0.001,
        lr_decay_steps=400e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=KnownBounds(-1, 1),
        device=torch.device("cuda"))