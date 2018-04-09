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

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

    def choose_action(self,actions,learning_rate=0.01,reward_decay=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

    def save_train_parameter(self,*args,**kwargs):
        pass

    def load_train_parameter(self,*args,**kwargs):
        pass

    def learn(self,*args,**kwargs):
        pass

    def train(self,*args,**kwargs):
        pass

class QTableAgent(Agent):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QTableAgent,self).__init__(actions,learning_rate,reward_decay)
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame()

    def check_state_exist(self, state):
        if tuple(state) not in self.q_table.columns:
            # append new state to q table
            self.q_table[tuple(state)] = [0]*len(self.actions)

    def choose_action(self, observation):
        observation = tuple(observation)
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table[observation]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.argmax()
        else:
            action = np.random.choice(self.actions)

        return action

    def save_train_parameter(self, name):
        self.q_table.to_pickle(name)

    def load_train_parameter(self, name):
        self.q_table = pd.read_pickle(name)

    def create_q_table(self,name):
        if os.path.exists(name):
            self.load_train_parameter(name)
        else:
            self.q_table = pd.DataFrame(index=self.actions)

class QLearningAgent(QTableAgent):

    def learn(self, s, a, r, s_ , done):
        s = tuple(s)
        s_ = tuple(s_)
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        if not done:
            q_target = r + self.reward_decay * self.q_table[s_].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table[s][a] += self.learning_rate * (q_target - q_predict)  # update

    def train(self,env,max_iterator = 1000):
        self.load_train_parameter("q_table.pkl")
        for episode in range(max_iterator):
            # initial observation
            observation = env.reset()
            step = 0

            while True:

                # fresh env
                env.render()

                # RL choose action based on observation
                action = self.choose_action(observation)

                # RL take action and get next observation and reward
                observation_, reward, done = env.step(observation, action, step)

                # RL learn from this transition
                self.learn(observation, action, reward, observation_, done)

                # swap observation
                observation = observation_
                step += 1
                # break while loop when end of this episode
                if done:
                    print(self.q_table)
                    self.save_train_parameter("q_table.pkl")
                    break
        # end of game
        print('game over')
        env.destroy()

class SarsaAgent(QTableAgent):

    def learn(self, s, a, r, s_,a_,done):
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        if not done:
            q_target = r + self.reward_decay * self.q_table[s_][a_] # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table[s][a] += self.learning_rate*(q_target - q_predict)  # update

    def train(self,env,max_iterator):
        self.load_train_parameter("q_table.pkl")
        for episode in range(max_iterator):
            # initial observation
            observation = env.reset()
            # RL choose action based on observation
            action = self.choose_action(observation)
            step = 0
            while True:

                # fresh env
                env.render()

                # RL take action and get next observation and reward
                observation_, reward, done = env.step(action, step)

                # RL choose action based on observarion_
                action_ = self.choose_action(observation_)

                # RL learn from this transition
                self.learn(observation, action, reward, observation_, action_, done)

                # swap observation and action
                observation = observation_
                action = action_
                step += 1
                # break while loop when end of this episode
                if done:
                    print(reward)
                    print(self.q_table)
                    self.save_train_parameter("q_table.pkl")
                    break

        # end of game
        print('game over')
        env.destroy()

class DQNAgent(Agent):

    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,
                 initial_epsilon=1,final_epsilon=0.001,
                 observe_iteration = 5000,
                 explore_iteration = 10000,
                 save_iteration=100,
                 memory_size = 10000,train_batch = 100):
        super(DQNAgent,self).__init__(actions,learning_rate,reward_decay)
        self.action_num = len(self.actions)
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.observe_iteration = observe_iteration
        self.explore_iteration = explore_iteration
        self.save_iteration = save_iteration
        self.memory_size = memory_size
        self.train_batch = train_batch

        #主要是为了神经网络的训练
        self.lb = LabelBinarizer()
        self.lb.fit([0,1,2,3])
        self.D = deque()
        # self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.ctreate_NN(input_size=4,output_size=self.action_num)

    def ctreate_NN(self,input_size,output_size):
        self.observation_holder = tf.placeholder("float", [None, input_size])
        self.output_holder = tf.placeholder("float", [None])
        self.action_holder = tf.placeholder("float", [None,output_size])

        # 三层的神经网络
        W = {
            "W1":tf.Variable(tf.random_normal([input_size,40])),
            "W2":tf.Variable(tf.random_normal([40,40])),
            "W3":tf.Variable(tf.random_normal([40,output_size])),
        }
        b = {
            "b1":tf.Variable(tf.random_normal([40])),
            "b2":tf.Variable(tf.random_normal([40])),
            "b3":tf.Variable(tf.random_normal([self.action_num])),
        }
        layer1 = tf.nn.relu(tf.matmul(self.observation_holder,W["W1"])+b["b1"])
        layer2 = tf.nn.relu(tf.matmul(layer1,W["W2"])+b["b2"])
        self.Q_s_network = tf.matmul(layer2,W["W3"])+b["b3"]
        # 定义损失函数
        Q_s_a = tf.reduce_sum(tf.multiply(self.Q_s_network, self.action_holder), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(self.output_holder - Q_s_a))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    def choose_action(self, observation, epsilon):
        # 利用神经网络得到当前的Q(observation_holder,:)
        Q_s = self.sess.run(self.Q_s_network,feed_dict={self.observation_holder: [observation]})
        # 贪婪的选取动作
        if np.random.random() <= epsilon:
            action = np.random.choice(self.actions)
        else:
            action = np.argmax(Q_s)
        return action

    def reduce_epsilon(self,t,epsilon):
        if epsilon > self.final_epsilon and t > self.observe_iteration:
            epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore_iteration
        return epsilon

    def load_train_parameter(self,name="saved_networks"):
        pass
        # 导入参数继续前面的训练
        # checkpoint = tf.train.get_checkpoint_state(name)
        # if checkpoint and checkpoint.model_checkpoint_path:
        #     self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
        #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
        # else:
        #     print("Could not find old network weights")

    def save_train_parameter(self,iteration,name ="saved_networks"):
        pass
        # self.saver.save(self.sess, name+ '/dqn', global_step=iteration)

    def save_to_memory(self,observation,action,reward,observation_,done):
        self.D.append((observation, action, reward, observation_,done))
        if len(self.D) > self.memory_size:
            self.D.popleft()

    def print_info(self,t,epsilon,action,reward):
        status = ""
        if t <= self.observe_iteration:
            status = "observe"
        elif t > self.observe_iteration and t < self.observe_iteration + self.explore_iteration:
            status = "explore"
        else:
            status = "train"

        print("TIMESTEP, {0}, / STATE, {1}, \
              / EPSILON, {2}, / ACTION, {3}, / \
              REWARD, {4}".format(t,status,epsilon,action,reward))

    def learn(self):
        # 选取小部分进行学习
        mini_batch = random.sample(self.D,self.train_batch)
        observation_batch = [d[0] for d in mini_batch]
        action_batch = [d[1] for d in mini_batch]
        reward_batch = [d[2] for d in mini_batch]
        observation_batch_ = [d[3] for d in mini_batch]
        #将单个数字的转换成为二进制数组形式
        self.lb.transform(action_batch)
        action_batch = self.lb.transform(action_batch)
        Q_s_ = self.sess.run(self.Q_s_network,feed_dict={self.observation_holder: observation_batch_})
        output_batch = []
        for i in range(0, len(mini_batch)):
            done = mini_batch[i][4]
            if done:
                output_batch.append(reward_batch[i])
            else:
                output_batch.append(reward_batch[i]+self.reward_decay*np.max(Q_s_[i]))

        # 利用梯度函数进行学习
        self.sess.run(self.train_step,feed_dict={
            self.output_holder: output_batch,
            self.action_holder: action_batch,
            self.observation_holder: observation_batch}
        )

    def train(self, env):

        # 初始化神经网络的参数
        self.sess.run(tf.initialize_all_variables())
        self.D.clear()

        # 导入数据
        self.load_train_parameter()
        # 初始化此时
        observation = env.reset()
        # 开始训练
        epsilon = self.initial_epsilon
        t = 0
        for iteration in range(1000000):

            #贪婪选取动作
            action = self.choose_action(observation,epsilon)

            #贪婪比率开始不断变大
            epsilon = self.reduce_epsilon(iteration,epsilon)

            # 执行选择的步骤
            observation_, reward, done = env.step(observation, action,t)

            # 存储用于后面的神经网络的训练
            self.save_to_memory(observation,action,reward,observation_,done)

            # 在观察期后开始训练
            if iteration > self.observe_iteration:
                self.learn()

            # 将观测值进行替换
            observation = observation_
            t += 1

            if done:
                t = 0
            # 每相隔一定的步数进行存储
            if iteration % self.save_iteration == 0:
                self.save_train_parameter(iteration)

            # 打印每一步的情况
            self.print_info(iteration,epsilon,action,reward)

