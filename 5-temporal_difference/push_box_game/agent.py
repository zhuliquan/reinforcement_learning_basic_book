#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2017/10/4

import numpy as np
from sklearn.preprocessing import LabelBinarizer
import cv2
import random
import pandas as pd
import os
import tensorflow as tf
from collections import deque


class Agent(object):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame()

    def check_state_exist(self, state):
        if tuple(state) not in self.q_table.columns:
            self.q_table[tuple(state)] = [0]*len(self.actions)

    def choose_action(self, observation):
        observation = tuple(observation)
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table[observation]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some action_space have same value
            action_idx = state_action.argmax()
        else:
            action_idx = np.random.choice(range(len(self.actions)))
        return action_idx

    def save_train_parameter(self, name):
        self.q_table.to_pickle(name)

    def load_train_parameter(self, name):
        self.q_table = pd.read_pickle(name)

    def learn(self,*args,**kwargs):
        pass

    def train(self,*args,**kwargs):
        pass

class QLearningAgent(Agent):

    def learn(self, s, a, r, s_ , done):
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        if not done:
            q_target = r + self.reward_decay * self.q_table[s_].max()  # next __state is not terminal
        else:
            q_target = r  # next __state is terminal
        self.q_table[s][a] += self.learning_rate * (q_target - q_predict)  # update

    def train(self,env,max_iterator=100):
        self.load_train_parameter("q_table.pkl")
        for episode in range(max_iterator):

            observation = env.reset()

            while True:
                # env.render()

                action_idx = self.choose_action(observation)

                observation_, reward, done,_ = env.step(self.actions[action_idx])

                print(observation,reward)

                self.learn(observation, action_idx, reward, observation_, done)

                observation = observation_

                if done:
                    self.save_train_parameter("q_table.pkl")
                    break

class SarsaAgent(Agent):

    def learn(self, s, a, r, s_,a_,done):
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        if not done:
            q_target = r + self.reward_decay * self.q_table[s_][a_] # next __state is not terminal
        else:
            q_target = r  # next __state is terminal
        self.q_table[s][a] += self.learning_rate*(q_target - q_predict)  # update

    def train(self,env,max_iterator=100):
        self.load_train_parameter("q_table.pkl")
        for episode in range(max_iterator):

            observation = env.reset()
            action_idx = self.choose_action(observation)
            while True:
                # env.render()
                print(observation)

                observation_, reward, done,_ = env.step(self.actions[action_idx])

                action_idx_ = self.choose_action(observation_)

                self.learn(observation, action_idx, reward, observation_, action_idx_, done)

                observation = observation_
                action_idx = action_idx_

                if done:
                    self.save_train_parameter("q_table.pkl")
                    break
