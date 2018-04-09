#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2017/10/4

import sys
sys.path.append("../push_box_game")
from env import  Maze
from agent import QLearningAgent

if __name__ == "__main__":
    env = Maze()
    RL = QLearningAgent(actions=list(range(env.n_actions)))
    env.after(100, RL.train(env))
    env.mainloop()