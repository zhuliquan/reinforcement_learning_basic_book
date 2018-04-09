#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2018/4/7
import gym
from gym.utils import seeding
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MazeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.map = np.array([
             [0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 1, 1],
             [1, 0, 0, 0, 0]],dtype=np.bool)

        self.states = [tuple(s) for s in np.argwhere(self.map == False)]
        self.walls  = [tuple(s) for s in np.argwhere(self.map == True)]
        self.terminate_states = ((4,2),)  # 终止状态为字典格式

        # 状态转移的数据格式为字典
        self.actions = ('n', 'e', 's', 'w')
        self.t = pd.DataFrame(data=None, index=self.states, columns=self.actions)
        self._trans_make()
        self.gamma = 0.8  # 折扣因子
        self.viewer = None
        self.state = None
        self.seed()

    def _trans_make(self):
        for s in self.states:
            for a in self.actions:
                n_s = np.array(s)
                if a == "n":
                    n_s += np.array([0,-1])
                elif a == "e":
                    n_s += np.array([1,0])
                elif a == "s":
                    n_s += np.array([0,1])
                elif a == "w":
                    n_s += np.array([-1,0])
                if (0 <= n_s).all() and (n_s <= 4).all() and not self.map[n_s[0],n_s[1]]:
                    self.t.loc[s,a] = tuple(n_s)

    def _reward(self, state, action):
        r = 0.0
        n_s = state
        if action == "n":
            n_s += np.array([0, -1])
        elif action == "e":
            n_s += np.array([1, 0])
        elif action == "s":
            n_s += np.array([0, 1])
        elif action == "w":
            n_s += np.array([-1, 0])
        if (0 <= n_s).all() and (n_s <= 4).all() and \
            n_s in self.terminate_states:
            r = 1.0
        return r

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer: self.viewer.close()


    def step(self, action):
        # 系统当前状态
        state = self.state
        if state in self.terminate_states:
            return state, 0, True, {}
        # 状态转移
        if pd.isna(env.t.loc[state, action]):
            next_state = state
        else:
            next_state = self.t.loc[state, action]
        self.state = next_state

        is_terminal = False
        if next_state in self.terminate_states:
            is_terminal = True

        r = self._reward(self.state, action)

        return next_state, r, is_terminal, {}

    def reset(self):
        self.state = self.states[np.random.choice(len(self.states))]
        return self.state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        unit = 50
        screen_width = 5 * unit
        screen_height = 5 * unit

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            #创建网格
            for c in range(5):
                line = rendering.Line((0,c*unit),(screen_width,c*unit))
                line.set_color(0,0,0)
                self.viewer.add_geom(line)
            for r in range(5):
                line = rendering.Line((r*unit, 0), (r*unit, screen_height))
                line.set_color(0, 0, 0)
                self.viewer.add_geom(line)

            # 创建墙壁
            for x,y in self.walls:
                r = rendering.make_polygon(
                    v=[[x * unit, y * unit],
                       [(x + 1) * unit, y * unit],
                       [(x + 1) * unit, (y + 1) * unit],
                       [x * unit, (y + 1) * unit],
                       [x * unit, y * unit]])
                r.set_color(0,0,0)
                self.viewer.add_geom(r)

            # 创建机器人
            self.robot = rendering.make_circle(20)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.6, 0.4)
            self.viewer.add_geom(self.robot)

            # 创建出口
            self.exit = rendering.make_circle(20)
            self.exitrans = rendering.Transform(translation=(4*unit+unit/2,2*unit+unit/2))
            self.exit.add_attr(self.exitrans)
            self.exit.set_color(0,1,0)
            self.viewer.add_geom(self.exit)

        if self.state is None:
            return None

        self.robotrans.set_translation(self.state[0]*unit + unit/2, self.state[1]*unit + unit/2)
        return self.viewer.render(return_rgb_array= mode == 'rgb_array')

if __name__ == '__main__':
    env = MazeEnv()
    env.reset()
    env.render()
