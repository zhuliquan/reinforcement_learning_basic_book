#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2018/4/2
import numpy as np
import pandas as pd
def main():
    S = ("S1","S2","S3","S4","S5")
    A = ("玩","退出","学习","发表","睡觉")
    strategy = pd.DataFrame(data=None,index=S,columns=A)
    reward = pd.DataFrame(data=None,index=S,columns=A)
    gama = 1
    strategy.loc["S1",:] = np.array([0.9,0.1,0,0,0])
    strategy.loc["S2",:] = np.array([0.5,0,0.5,0,0])
    strategy.loc["S3",:] = np.array([0,0,0.8,0,0.2])
    strategy.loc["S4",:] = np.array([0,0,0,0.4,0.6])
    strategy.loc["S5",:] = np.array([0,0,0,0,0])
    print("策略")
    print(strategy)
    reward.loc["S1", :] = np.array([-1, 0,   0,  0, 0])
    reward.loc["S2", :] = np.array([-1, 0,  -2,  0, 0])
    reward.loc["S3", :] = np.array([0,  0,  -2,  0, 0])
    reward.loc["S4", :] = np.array([0,  0,   0, 10, 1])
    reward.loc["S5", :] = np.array([0, 0, 0, 0, 0])
    print("回报函数")
    print(reward)
if __name__ == '__main__':
    main()