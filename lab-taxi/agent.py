import numpy as np
from collections import defaultdict
from math import floor
import random


def randargmax(nparray):
    return np.random.choice(np.flatnonzero(nparray == nparray.max()))


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] += 1


class Sarsa(object):

    def __init__(self, nA, alpha=0.1, alpha_decay=1.0, alpha_min=0.001,
                 gamma=1.0, epsilon=0.3, eps_decay=0.99, eps_min=0.05):
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha  # learning rate
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.gamma = gamma  # time discount
        self.epsilon = epsilon  # epsilon in epsilon-greedy
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.n_episodes = 0

    def select_action(self, state):
        if self.epsilon > 0.000001:
            action = min(floor(np.random.rand() / (self.epsilon / self.nA)), self.nA)
            if action == self.nA:  # select greedy action
                action = randargmax(self.Q[state])
        else:
            action = randargmax(self.Q[state])
        return action

    def step(self, state, action, reward, next_state, done):
        next_action = self.select_action(next_state)
        q_old = self.Q[state][action]
        q_est = reward + self.gamma * self.Q[next_state][next_action]
        self.Q[state][action] += self.alpha * (q_est - q_old)

        self.n_episodes += 1
        if done:
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
            self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)


class QLearning(object):

    def __init__(self, nA, alpha=0.1, alpha_decay=1.0, alpha_min=0.001,
                 gamma=1.0, epsilon=1.0, eps_decay=0.999, eps_min=0.01):
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha  # learning rate
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.gamma = gamma  # time discount
        self.epsilon = epsilon  # epsilon in epsilon-greedy
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.n_episodes = 0

    def select_action(self, state):
        if self.epsilon > 0.000001:
            action = min(floor(np.random.rand() / (self.epsilon / self.nA)), self.nA)
            if action == self.nA:  # select greedy action
                action = randargmax(self.Q[state])
        else:
            action = randargmax(self.Q[state])
        return action

    def step(self, state, action, reward, next_state, done):
        q_old = self.Q[state][action]
        q_est = reward + self.gamma * max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (q_est - q_old)

        self.n_episodes += 1
        if done:
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
            self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)


class ExpectedSarsa(object):

    def __init__(self, nA, alpha=0.1, alpha_decay=1.0, alpha_min=0.001,
                 gamma=1.0, epsilon=1.0, eps_decay=0.999, eps_min=0.01):
        self.nA = nA
        self.Q = defaultdict(lambda: np.full(self.nA, 0.0))
        self.alpha = alpha  # learning rate
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.gamma = gamma  # time discount
        self.epsilon = epsilon  # epsilon in epsilon-greedy
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.n_episodes = 0

    def select_action(self, state):
        if self.epsilon > 0.000001:
            action = floor(np.random.rand() / (self.epsilon / self.nA))
            if action >= self.nA:  # select greedy action
                action = randargmax(self.Q[state])
        else:
            action = randargmax(self.Q[state])
        return action

    def step(self, state, action, reward, next_state, done):
        p = np.full(self.nA, self.epsilon / self.nA)
        p[randargmax(self.Q[next_state])] += 1 - self.epsilon

        q_old = self.Q[state][action]
        q_est = reward + self.gamma * np.dot(p, self.Q[next_state])
        self.Q[state][action] += self.alpha * (q_est - q_old)

        self.n_episodes += 1
        if done:
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
            self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)
