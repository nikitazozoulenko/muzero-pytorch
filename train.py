import os
from typing import Dict, List, Optional, NamedTuple
import numpy as np
import torch
import torch.nn as nn

from config import MuZeroConfig, make_board_game_config
from network import Network
from environment import Game
from mcts import play_game

class ReplayBuffer(object):

    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.num_unroll_steps = config.num_unroll_steps
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(i, self.config.device), g.history[i:i + num_unroll_steps],
                g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
                for (g, i) in game_pos]

    def sample_game(self) -> Game:
        # Sample game from buffer either uniformly or according to some priority.
        return self.buffer[np.random.randint(len(self.buffer))]

    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        return np.random.randint(len(game.history)-self.num_unroll_steps)


def save_network(network, i, path="save_dir/"):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(network.to("cpu").state_dict(), path+"model"+str(i)+".pth")
    network.to(network.device)


def train_network(config: MuZeroConfig, network):
    replay_buffer = ReplayBuffer(config)
    LogSoftmax = nn.LogSoftmax(dim=1).to(network.device)
    MSE = nn.MSELoss().to(network.device)
    optimizer = torch.optim.SGD(network.parameters(), config.lr_init, momentum=0.9, weight_decay=config.weight_decay)

    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            save_network(network, i)
            print("saving")
        print(i)
        network.eval()
        game = play_game(config, network)
        replay_buffer.save_game(game)
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        network.train()
        update_weights(config, network, batch, LogSoftmax, MSE, optimizer)
    save_network(network, config.training_steps)


def update_weights(config, network, batch, LogSoftmax, MSE, optimizer):
    observations = []
    actions = []
    values = []
    rewards = []
    policies = []
    for obs, a, target in batch:
        observations.append(obs)
        actions.append(torch.tensor(a))
        v_list = []
        r_list = []
        p_list = []
        for v, r, p in target:
            v_list.append(v)
            r_list.append(r)
            p_list.append(torch.tensor(p))
        values.append(torch.tensor(v_list))
        rewards.append(torch.tensor(r_list))
        policies.append(torch.stack(p_list, dim=0))
    observations = torch.cat(observations, dim=0)
    actions = torch.stack(actions, dim=1).to(network.device)
    values = torch.stack(values, dim=1).float().to(network.device)
    rewards = torch.stack(rewards, dim=1).float().to(network.device)
    policies = torch.stack(policies, dim=1).float().to(network.device)

    #is now in correct format: loss below now
    loss = 0
    v, r, p, s = network.initial_inference(observations)
    loss += MSE(v, values[0])
    loss += MSE(r, rewards[0])
    loss -= (LogSoftmax(p) * policies[0]).sum()
    for i in range(1, config.num_unroll_steps+1):
        v, r, p, s = network.recurrent_inference(s, actions[i-1])
        loss += MSE(v, values[i]) / config.num_unroll_steps
        loss += MSE(r, rewards[i]) / config.num_unroll_steps
        loss -= (LogSoftmax(p) * policies[i]).sum() / config.num_unroll_steps
        s.register_hook(lambda grad: grad * 0.5)
    loss = loss / config.batch_size
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    config = make_board_game_config()
    network = Network(config).to(config.device)
    train_network(config, network)