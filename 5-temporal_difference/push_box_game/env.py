#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2017/10/4
"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [_reward = -1].
Yellow bin circle:      paradise    [_reward = +1].
All other observation_space:       ground      [_reward = 0].
This script is the environment part of this example. The RL is in agent.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 5  # grid height
MAZE_W = 5  # grid width
MAX_ITERATOR = 1000
WALL_POSITION = [(0, 1), (0, 3), (2, 1), (2, 2)]
BOX_POSITION = np.array((2,3))
WORKER_POSITION = np.array((3,3))
DEST_POSITION = np.array((2,0))
BASE_ACTION = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
REWARD = [
    [-100,-100, 100,   0,-100],
    [-100,-100,-100,   0,-100],
    [-100,-100,-100,   0,-100],
    [-100,-100,   0,   0,-100],
    [-100,-100,-100,-100,-100],
]

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.max_iterator = MAX_ITERATOR
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self.box_step = 0
        self.worker_step = 0
        self.box_position = BOX_POSITION
        self.worker_position = WORKER_POSITION
        self.dest_position = DEST_POSITION
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # 绘制网格
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        #创造墙壁
        for x,y in WALL_POSITION:
            self.canvas.create_rectangle(
                x*UNIT,y*UNIT,
                (x+1)*UNIT,(y+1)*UNIT,
                fill = "black"
            )
        #创造目标
        self.canvas.create_oval(
            DEST_POSITION[0]*UNIT,DEST_POSITION[1]*UNIT,
            (DEST_POSITION[0]+1)*UNIT,(DEST_POSITION[1]+1)*UNIT,
            fill = "blue"
        )

        # 初始化箱子与人
        self.box_position = np.array(BOX_POSITION)
        self.worker_position = np.array(WORKER_POSITION)
        self.box_rect = self.canvas.create_rectangle(
            BOX_POSITION[0] * UNIT,
            BOX_POSITION[1] * UNIT,
            (BOX_POSITION[0] + 1) * UNIT,
            (BOX_POSITION[1] + 1) * UNIT,
            fill="red"
        )
        self.worker_rect = self.canvas.create_rectangle(
            WORKER_POSITION[0] * UNIT,
            WORKER_POSITION[1] * UNIT,
            (WORKER_POSITION[0] + 1) * UNIT,
            (WORKER_POSITION[1] + 1) * UNIT,
            fill="green"
        )
        # canvas打包所有
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        #删除状态相关的东西
        self.canvas.delete(self.box_rect)
        self.canvas.delete(self.worker_rect)
        #初始化箱子与人
        self.box_position = np.array(BOX_POSITION)
        self.worker_position = np.array(WORKER_POSITION)
        self.box_rect = self.canvas.create_rectangle(
            BOX_POSITION[0] * UNIT,
            BOX_POSITION[1] * UNIT,
            (BOX_POSITION[0] + 1) * UNIT,
            (BOX_POSITION[1] + 1) * UNIT,
            fill="red"
        )
        self.worker_rect = self.canvas.create_rectangle(
            WORKER_POSITION[0] * UNIT,
            WORKER_POSITION[1]*UNIT,
            (WORKER_POSITION[0] + 1) * UNIT,
            (WORKER_POSITION[1] + 1) * UNIT,
            fill="green"
        )
        #箱子走的步数为0
        self.box_step = 0
        self.worker_step = 0
        # 返回观察值
        return np.concatenate((self.box_position,self.worker_position))

    def move_ok(self, pos):
        flag = False
        if tuple(pos) not in WALL_POSITION and \
           0 <= pos[0] < MAZE_W and 0 <= pos[1] < MAZE_H:
            flag = True
        return flag

    def get_reward(self,s, s_,t):

        reward = REWARD[s_[1]][s_[0]]
        if (s_[0:2] == self.dest_position).all() and self.step != 0:
            # 由于箱子用的路程最短所形回报加成
            reward /= (self.box_step + self.worker_step)
            done = True
        elif reward < 0 or t == self.max_iterator:
            done = True
        else:
            done = False
        #表示箱子有移动的情况
        if (s[0:2] != s_[0:2]).all():
            if reward >= 0:
                #说明推动带来了好的预期
                reward += 2
            elif reward < 0:
                #说明推动情况更加糟糕了
                reward -= 3
        return reward,done

    def step(self, s, action,t):
        box_action = np.array([0,0])
        worker_action = np.array([0,0])
        next_worker_position = self.worker_position + BASE_ACTION[action, :]
        #判断人可不可以移动
        if self.move_ok(next_worker_position):
            self.worker_step += 1
            self.worker_position = next_worker_position
            worker_action = BASE_ACTION[action, :]
            #判断箱子可以移动
            if (next_worker_position == self.box_position).all():

                next_box_postion = self.box_position + BASE_ACTION[action, :]
                if self.move_ok(next_box_postion):

                    #说明箱子可以移动
                    self.box_step += 1
                    box_action = BASE_ACTION[action, :]
                    self.box_position = next_box_postion
                else:

                    #虽然箱子在前方但是只是挡住了自己的路线,需要将已经移动的人还原
                    self.worker_step -= 1
                    worker_action = np.array([0,0])
                    self.worker_position -= BASE_ACTION[action, :]

        #得到下一步的观测值
        s_ = np.concatenate((self.box_position,self.worker_position))
        #在动画里面移动
        self.canvas.move(self.box_rect, box_action[0] * UNIT, box_action[1] * UNIT)
        self.canvas.move(self.worker_rect, worker_action[0] * UNIT, worker_action[1] * UNIT)

        #观察下一步的情况情况
        reward,done = self.get_reward(s, s_,t)

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()

def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a,0)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()