from typing import Dict, List, Optional, NamedTuple
import numpy as np
import torch
import torch.nn as nn

class Player(object):
    pass


class ActionHistory(object):
    """Simple history container used inside the search.

    Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[int], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: int):
        self.history.append(action)

    def last_action(self) -> int:
        return self.history[-1]

    def action_space(self) -> List[int]:
        return [int(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        return Player()



class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, action_space_size: int, discount: float):
        #environment
        self.size = 3
        self.win_length = 3
        self.board_history = [np.zeros(((self.size, self.size)), dtype=np.float32)]
        self.turn_val = 1 # 1 and -1
        self.terminal=False

        #model things
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount

    def step(self, action): #assume we already have a valid action
        next_board = self.board_history[-1].copy()
        next_board[action//self.size, action%self.size] = self.turn_val
        self.board_history.append(next_board)
        self.terminal, coords = self.check_win()
        if self.terminal:
            if coords:
                r = self.turn_val # 1 cross, -1 circle
            else:
                r = 0
        else:
            r = 0
        self.turn_val *=-1
        return r



    def check_win(self):
        # Game specific termination rules.
        '''Checks if any player has won and returns list [True/False, coords]'''
        terminate, coords = self._check_win_for_white(self.board_history[-1])
        if terminate:
            return terminate, coords
        terminate, coords = self._check_win_for_white(-1 * self.board_history[-1])
        return terminate, coords


    def _check_win_for_white(self, board):
        '''Finds the positions of the win for "X" (player_val=1). 
           Returns tuple (terminate, coord_list), empty list if no win, else list (y,x) coords of positions.'''
        n_cols = np.zeros((self.size)) # |
        n_rows = np.zeros((self.size)) # --
        n_diag = np.zeros((2*self.size-1)) # \
        n_anti_diag = np.zeros((2*self.size-1)) # /
        non_zeros = 0

        #enumerate the amount of Xs
        for y, rows in enumerate(board):
            for x, value in enumerate(rows):
                if value != 0:
                    non_zeros += 1
                if value == 1:
                    n_cols[x] += 1
                    n_rows[y] += 1
                    n_diag[self.size-1-y+x] += 1
                    n_anti_diag[x+y] += 1
        
        #find coords for lists with enough Xs
        for (v,  kind) in [(n_cols, "cols"), (n_rows, "rows"), (n_diag, "diag"), (n_anti_diag, "anti_diag")]:
            for i, val in enumerate(v):
                if val >= self.win_length:
                    coords = self._find_coords(kind, i, board)
                    if coords:
                        return True, coords

        #look for draw
        if non_zeros == self.size**2:
            return True, []

        return False, []

        
    def _find_coords(self, kind, index, board):
        '''Returns a list of (y, x) coordinates of a vector where there exists a win'''
        coords = []
        count = 0

        if kind == "cols":
            for i, val in enumerate(board[:, index]):
                if val == 1:
                    count += 1
                    coords += [(i, index)]
                    if count == self.win_length:
                        return coords
                else:
                    count = 0
                    coords = []
        
        elif kind == "rows":
            for i, val in enumerate(board[index, :]):
                if val == 1:
                    count += 1
                    coords += [(index, i)]
                    if count == self.win_length:
                        return coords
                else:
                    count = 0
                    coords = []
        
        elif kind == "diag":
            for i in range(self.size-abs(index-self.size+1)):
                if index <= self.size-1:
                    coord = (self.size-1-i, self.size-1-abs(index-self.size+1)-i)
                else: #if index > self.size-1
                    coord = (self.size-1-(i+abs(index-self.size+1)), self.size-1-i)
                if board[coord] == 1:
                    count += 1
                    coords += [coord]
                    if count == self.win_length:
                        return coords
                else:
                    count = 0
                    coords = []

        elif kind == "anti_diag":
            for i in range(self.size-abs(index-self.size+1)):
                if index <= self.size-1:
                    coord = (i, self.size-1-abs(index-self.size+1)-i)
                else: #if index > self.size-1
                    coord = (i+abs(index-self.size+1), self.size-1-i)
                if board[coord] == 1:
                    count += 1
                    coords += [coord]
                    if count == self.win_length:
                        return coords
                else:
                    count = 0
                    coords = []
        return []

    def legal_actions(self) -> List[int]:
        # Game specific calculation of legal actions.
        allowed = (self.board_history[-1].reshape(-1) == 0).astype(np.uint8)
        return np.nonzero(allowed)[0]


    def apply(self, action: int):
        reward = self.step(action)
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (int(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_image(self, state_index: int, device):
        # Game specific feature planes.
        tensor = torch.from_numpy(self.board_history[state_index]).to(device)
        return tensor.view(1, 1, self.size, self.size)


    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int,
                    to_play: Player):
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount**td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i  # pytype: disable=unsupported-operands

            if current_index < len(self.root_values):
                targets.append((value, self.rewards[current_index],
                                self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, 0, []))
        return targets

    def to_play(self) -> Player:
        return Player()

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)
