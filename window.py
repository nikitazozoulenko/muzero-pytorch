import time

from tkinter import Tk, Canvas, Text, BOTH, W, N, E, S, messagebox
from tkinter.ttk import Frame, Button, Entry, Label, Style
import numpy as np
import torch
import torch.nn as nn

from environment import Game
from mcts import run_mcts, select_action, Node, expand_node, add_exploration_noise
from config import make_board_game_config
from network import Network, NetworkOutput



def create_button(master, row, col, text, func=None, pady=7, padx=3):
    '''Takes in master, row number, col number, button display text, callback 
       function, and position padding and returns the created button widget'''
    button = Button(master, text=text, command=func, width=7)
    button.grid(row=row, column=col, pady=pady, padx=padx)
    return button


def create_label(master, row, col, text, pady=0, padx=2):
    '''Takes in master, row number, col number, label display text, and 
       position padding and returns the created labe widgetl'''
    label = Label(master, text=text)
    label.grid(row=row, column=col, pady=pady, padx=padx, sticky=W)
    return label


def create_entry(master, row, col, text, pady=0, padx=0):
    '''Takes in master, row number, col number, entry default value, and 
       position padding and returns the created entry widget'''
    entry = Entry(master)
    entry.insert(0, text)
    entry.grid(row=row, column=col, pady=pady, padx=padx)
    entry.config({"width":5})
    return entry


class Window(Frame):
    def __init__(self, tk_root, network):
        super().__init__()
        self.network = network.eval()
        self.config = network.config
        self.board_size = int(np.sqrt(self.config.action_space_size))
        self.game = self.config.new_game()

        self.tk_root = tk_root
        self.draw_area = DrawArea(self, self.game, self.board_size)
        self.initUI()

    
    def initUI(self):
        '''Initializes the graphical interface through specified button,
           label, and entry specs'''
        self.master.title("Conway's Game of Life")
        self.pack(fill=BOTH, expand=True)

        btn_specs = [{"text": " Set\nRules", "func":self._rules_callback},
                     {"text": "Reset", "func":self._reset_callback},
                     {"text": "Step", "func":self._step_callback}]
        lbl_specs = [{"text": "max MCTS steps"},
                     {"text": "Update game rules"},
                     {"text": "Reset game to empty board"},
                     {"text": "Let the agent take a step"}]
        entry_specs = [{"text": "?"}]
        self.labels = [create_label(self, row=j, col=4, text=spec["text"]) for j, spec in enumerate(lbl_specs)]
        self.entries = [create_entry(self, row=j, col=3, text=spec["text"]) for j, spec in enumerate(entry_specs)]
        self.buttons = [create_button(self, row=j+len(entry_specs), col=3, text=spec["text"], func=spec["func"]) for j, spec in enumerate(btn_specs)]


    def _step_callback(self):
        '''Callback function for button that steps through the game world'''
        if not self.game.terminal:
            root = Node(0)
            current_observation = self.game.make_image(-1, self.network.device)
            expand_node(root, self.game.to_play(), self.game.legal_actions(),
                        self.network.initial_inference(current_observation))
            add_exploration_noise(self.config, root)

            # We then run a Monte Carlo Tree Search using only action sequences and the
            # model learned by the network.
            run_mcts(self.config, root, self.game.action_history(), self.network)
            #action = select_action(self.config, len(self.game.history), root)
            action = select_action(self.config, 9, root)
            self.game.apply(action)
            self.game.store_search_statistics(root)
        self.draw_area.draw()
    

    def _reset_callback(self):
        '''Callback function for button that resets board and mcts'''
        self.game = self.config.new_game()
        self.draw_area.game = self.game
        self.draw_area.draw()
        print("reset")


    def _rules_callback(self):
        '''Callback function for button that reads game rules through GUI and loads them'''
        # entries = [e.get() for e in self.entries]
        # try:
        #     entries = [float(entry) for entry in entries]
        # except Exception:
        #     messagebox.showerror(title="Entry Error", message="Entry has to be a valid number")
        # self.max_mcts_steps, self.mcts_eps = entries
        # self.draw_area.draw()
    


class DrawArea(Canvas):
    '''A drawing canvas which has acces to the game world 
       so that it can draw and change it through the user gui'''
    def __init__(self, master, game, board_size, size=500, border_thickness=4):
        super().__init__(master, width=size, height=size)
        self.game = game
        self.board_size = board_size
        self.size=size
        self.border_thickness = border_thickness

        self.config({"background":"black"})
        self.bind("<Button-1>", self.mb1_callback)
        self.grid(row=0, column=0, rowspan=9, padx=5, pady=5, sticky=N+S+W+E)
        self.draw()
    

    def mb1_callback(self, event):
        '''Callback that converts mouse click event coordinates to a change in game world'''
        #convert to indices
        n = self.board_size
        size = self.size
        border = self.border_thickness
        grid = (size-(n-1)*border)//n
        start = (size - n*grid - (n-1)*border)//2
        coord = (np.array([event.y, event.x]) - start)//(grid+border)

        #change value
        a = n*coord[0] + coord[1]
        if a in self.game.legal_actions():
            self.game.apply(a)
        self.draw()
        

    def draw(self):
        '''Draws game world on canvas'''
        self.delete("all")
        n = self.board_size
        size = self.size
        border = self.border_thickness
        game_board = self.game.board_history[-1]
        grid = (size-(n-1)*border)//n
        start = (size - n*grid - (n-1)*border)//2

        terminate, coords = self.game.check_win()
        color = np.array([["white"]*n]*n)
        for coord in coords:
            color[coord[0], coord[1]] = "blue"
        if terminate and not coords:
            color = np.array([["grey"]*n]*n)

        print(game_board)

        for y in range(n):
            for x in range(n):
                x0 = start+ x*(grid+border)
                y0 = start + y*(grid+border)
                x1 = x0+grid
                y1 = y0+grid
                if game_board[y, x] == 1:
                    self.create_rectangle(x0, y0, x1, y1, fill=color[y, x])
                    self.create_line(x0+10, y0+10, x1-10, y1-10, fill="black", width=10)
                    self.create_line(x0+10, y1-10, x1-10, y0+10, fill="black", width=10)
                elif game_board[y, x] == -1:
                    self.create_rectangle(x0, y0, x1, y1, fill=color[y, x])
                    self.create_oval(x0+10, y0+10, x1-10, y1-10, fill="black")
                else:
                    self.create_rectangle(x0, y0, x1, y1, fill=color[y, x])


if __name__ == '__main__':
    config = make_board_game_config()
    network = Network(config).to(config.device)
    network.load_state_dict(torch.load("save_dir/model16000.pth")) # needs more training
    tk_root = Tk()
    tk_root.geometry("900x513+450+100")
    app = Window(tk_root, network)
    tk_root.mainloop()
