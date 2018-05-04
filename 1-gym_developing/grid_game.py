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


class GridEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):

        self.observation_space = (1, 2, 3, 4, 5, 6, 7, 8)  # 状态空间
        self.x = [140, 220, 300, 380, 460, 140, 300, 460]
        self.y = [250, 250, 250, 250, 250, 150, 150, 150]
        self.__terminal_space = dict()  # 终止状态为字典格式
        self.__terminal_space[6] = 1
        self.__terminal_space[7] = 1
        self.__terminal_space[8] = 1

        # 状态转移的数据格式为字典
        self.action_space = ('n', 'e', 's', 'w')
        self.t = pd.DataFrame(data=None, index=self.observation_space, columns=self.action_space)
        self.t.loc[1, "s"] = 6
        self.t.loc[1, "e"] = 2
        self.t.loc[2, "w"] = 1
        self.t.loc[2, "e"] = 3
        self.t.loc[3, "s"] = 7
        self.t.loc[3, "w"] = 2
        self.t.loc[3, "e"] = 4
        self.t.loc[4, "w"] = 3
        self.t.loc[4, "e"] = 5
        self.t.loc[5, "s"] = 8
        self.t.loc[5, "w"] = 4
        self.__gamma = 0.8  # 折扣因子
        self.viewer = None
        self.__state = None
        self.seed()

    def _reward(self, state):
        r = 0.0
        if state in (6,8):
            r = -1.0
        elif state == 7:
            r = 1.0
        return r

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer: self.viewer.close()

    def transform(self,state,action):
        #卫语句
        if state in self.__terminal_space:
            return state, self._reward(state), True, {}

        # 状态转移
        if pd.isna(self.t.loc[state, action]):
            next_state = state
        else:
            next_state = self.t.loc[state, action]

        # 计算回报
        r = self._reward(next_state)

        # 判断是否终止
        is_terminal = False
        if next_state in self.__terminal_space:
            is_terminal = True

        return next_state, r, is_terminal, {}

    def step(self, action):
        state = self.__state

        next_state, r, is_terminal,_ = self.transform(state,action)

        self.__state = next_state

        return next_state, r, is_terminal, {}

    def reset(self):
        self.__state = np.random.choice(self.observation_space)
        return self.__state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # 创建网格世界
            self.line1 = rendering.Line((100, 300), (500, 300))
            self.line2 = rendering.Line((100, 200), (500, 200))
            self.line3 = rendering.Line((100, 300), (100, 100))
            self.line4 = rendering.Line((180, 300), (180, 100))
            self.line5 = rendering.Line((260, 300), (260, 100))
            self.line6 = rendering.Line((340, 300), (340, 100))
            self.line7 = rendering.Line((420, 300), (420, 100))
            self.line8 = rendering.Line((500, 300), (500, 100))
            self.line9 = rendering.Line((100, 100), (180, 100))
            self.line10 = rendering.Line((260, 100), (340, 100))
            self.line11 = rendering.Line((420, 100), (500, 100))
            # 创建第一个骷髅
            self.kulo1 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(140, 150))
            self.kulo1.add_attr(self.circletrans)
            self.kulo1.set_color(0, 0, 0)
            # 创建第二个骷髅
            self.kulo2 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(460, 150))
            self.kulo2.add_attr(self.circletrans)
            self.kulo2.set_color(0, 0, 0)
            # 创建金条
            self.gold = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(300, 150))
            self.gold.add_attr(self.circletrans)
            self.gold.set_color(1, 0.9, 0)
            # 创建机器人
            self.robot = rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.6, 0.4)

            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.kulo1)
            self.viewer.add_geom(self.kulo2)
            self.viewer.add_geom(self.gold)
            self.viewer.add_geom(self.robot)

        if self.__state is None:
            return None

        self.robotrans.set_translation(self.x[self.__state - 1], self.y[self.__state - 1])
        return self.viewer.render(return_rgb_array= mode == 'rgb_array')


if __name__ == '__main__':
    env = GridEnv()
    env.reset()
    env.render()