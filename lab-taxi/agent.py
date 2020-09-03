import numpy as np
from collections import defaultdict
from math import floor
import random
from gym import Env
from gym.envs.toy_text import TaxiEnv
from scipy.stats import norm


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def randargmax(nparray):
    return np.random.choice(np.flatnonzero(nparray == nparray.max()))


class Agent:

    def __init__(self, env=6):
        """
        Initialize agent.

        :param Env env: Open AI environment.
        """
        self.env = env
        self.nA = env.nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def act(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(self.nA)

    def learn(self, state, action, reward, next_state, done):
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


class Sarsa(Agent):

    def __init__(self, env, alpha=0.1, alpha_decay=1.0, alpha_min=0.001,
                 gamma=1.0, epsilon=0.3, eps_decay=0.99, eps_min=0.05):
        super().__init__(env)
        self.env = env
        self.nA = env.nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha  # learning rate
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.gamma = gamma  # time discount
        self.epsilon = epsilon  # epsilon in epsilon-greedy
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.n_episodes = 0

    def act(self, state):
        if self.epsilon > 0.000001:
            action = min(floor(np.random.rand() / (self.epsilon / self.nA)), self.nA)
            if action == self.nA:  # select greedy action
                action = randargmax(self.Q[state])
        else:
            action = randargmax(self.Q[state])
        return action

    def learn(self, state, action, reward, next_state, done):
        next_action = self.act(next_state)
        q_old = self.Q[state][action]
        q_est = reward + self.gamma * self.Q[next_state][next_action]
        self.Q[state][action] += self.alpha * (q_est - q_old)

        self.n_episodes += 1
        if done:
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
            self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)


class QLearning(Agent):

    def __init__(self, env, alpha=0.1, alpha_decay=1.0, alpha_min=0.001,
                 gamma=1.0, epsilon=1.0, eps_decay=0.999, eps_min=0.01):
        super().__init__(env)
        self.env = env
        self.nA = env.nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha  # learning rate
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.gamma = gamma  # time discount
        self.epsilon = epsilon  # epsilon in epsilon-greedy
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.n_episodes = 0

    def act(self, state):
        if self.epsilon > 0.000001:
            action = min(floor(np.random.rand() / (self.epsilon / self.nA)), self.nA)
            if action == self.nA:  # select greedy action
                action = randargmax(self.Q[state])
        else:
            action = randargmax(self.Q[state])
        return action

    def learn(self, state, action, reward, next_state, done):
        q_old = self.Q[state][action]
        q_est = reward + self.gamma * max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (q_est - q_old)

        self.n_episodes += 1
        if done:
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
            self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)


class ExpectedSarsa(Agent):

    def __init__(self, env, alpha=0.1, alpha_decay=1.0, alpha_min=0.001,
                 gamma=1.0, epsilon=1.0, eps_decay=0.999, eps_min=0.01):
        super().__init__(env)
        self.env = env
        self.nA = env.nA
        self.Q = defaultdict(lambda: np.full(self.nA, 0.0))
        self.alpha = alpha  # learning rate
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.gamma = gamma  # time discount
        self.epsilon = epsilon  # epsilon in epsilon-greedy
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.n_episodes = 0

    def act(self, state):
        if self.epsilon > 0.000001:
            action = floor(np.random.rand() / (self.epsilon / self.nA))
            if action >= self.nA:  # select greedy action
                action = randargmax(self.Q[state])
        else:
            action = randargmax(self.Q[state])
        return action

    def learn(self, state, action, reward, next_state, done):
        p = np.full(self.nA, self.epsilon / self.nA)
        p[randargmax(self.Q[next_state])] += 1 - self.epsilon

        q_old = self.Q[state][action]
        q_est = reward + self.gamma * np.dot(p, self.Q[next_state])
        self.Q[state][action] += self.alpha * (q_est - q_old)

        self.n_episodes += 1
        if done:
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
            self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)


class QLearningGuided(Agent):

    def __init__(self, env, alpha=0.1, alpha_decay=1.0, alpha_min=0.001,
                 gamma=1.0, epsilon=1.0, eps_decay=0.999, eps_min=0.01):
        """
        Initialize agent.

        :param TaxiEnv env: Open AI environment.
        """
        super().__init__(env)
        self.env = env
        self.nA = env.nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha  # learning rate
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.gamma = gamma  # time discount
        self.epsilon = epsilon  # epsilon in epsilon-greedy
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.n_episodes = 0

    def act(self, state):
        action_list = self.get_available_actions(state)
        return self.choose_action(state, action_list)

    def get_available_actions(self, state):
        available_actions = []
        taxi_row, taxi_col, pass_idx, dest_idx = self.env.decode(state)
        if self.env.desc[2 + taxi_row, 2 * taxi_col + 1] in [b" ", b"R", b"G", b"Y", b"B"]:
            available_actions.append(0)
        if self.env.desc[taxi_row, 2 * taxi_col + 1] in [b" ", b"R", b"G", b"Y", b"B"]:
            available_actions.append(1)
        if self.env.desc[1 + taxi_row, 2 * taxi_col + 2] == b":":
            available_actions.append(2)
        if self.env.desc[1 + taxi_row, 2 * taxi_col] == b":":
            available_actions.append(3)
        if pass_idx != 4 and ((taxi_row, taxi_col) == self.env.locs[pass_idx]):
            available_actions.append(4)
        if ((taxi_row, taxi_col) == self.env.locs[dest_idx]) and pass_idx == 4:
            available_actions.append(5)
        return available_actions

    def choose_action(self, state, action_list):
        nA = len(action_list)
        v = self.Q[state][action_list]
        if self.epsilon > 0.000001:
            action = min(floor(np.random.rand() / (self.epsilon / nA)), nA)
            if action == nA:  # select greedy action
                action = action_list[randargmax(v)]
            else:
                action = action_list[action]
        else:
            action = action_list[randargmax(v)]
        return action

    def learn(self, state, action, reward, next_state, done):
        q_old = self.Q[state][action]
        q_est = reward + self.gamma * max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (q_est - q_old)

        self.n_episodes += 1
        if done:
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
            self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)


class KG(Agent):

    def __init__(self, env, gamma=1.0, epsilon=10.0, eps_decay=0.999, eps_min=0.01, online=True):
        super().__init__(env)
        self.env = env
        self.nA = env.nA

        self.gamma = gamma
        self.epsilon = epsilon  # epsilon in epsilon-greedy
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.n_episodes = 0

        self.online = online

        self.mu_init = 0.0
        self.sigma_init = 10000
        self.sigma_measure = 1.0

        self.mu = defaultdict(lambda: np.ones([self.nA]) * self.mu_init)
        self.sigma = defaultdict(lambda: np.ones([self.nA]) * self.sigma_init)

        sigma_tilde_init = self.sigma_init / np.sqrt(1 + (self.sigma_measure / self.sigma_init) ** 2)
        self.sigma_tilde = defaultdict(lambda: np.ones([self.nA]) * sigma_tilde_init)
        self.zeta = defaultdict(lambda: np.zeros([self.nA]))

        self.kg_nu = keydefaultdict(lambda key: self.sigma_tilde[key] * (self.zeta[key] * norm.cdf(self.zeta[key]) + norm.pdf(self.zeta[key])))

    def act(self, state):
        if self.online:
            action = randargmax(self.mu[state] + self.epsilon * self.kg_nu[state])
        else:
            action = randargmax(self.kg_nu[state])
        return action

    def learn(self, state, action, reward, next_state, done):
        reward_est = reward + self.gamma * max(self.mu[next_state])
        # Mu updating for current action
        self.mu[state][action] = ((reward_est * self.sigma[state][action] ** 2) + (self.mu[state][action] * self.sigma_measure ** 2)) \
            / (self.sigma_measure ** 2 + self.sigma[state][action] ** 2)

        # Sigma updating for current action
        self.sigma[state][action] = (self.sigma[state][action] * self.sigma_measure) \
            / (np.sqrt(self.sigma[state][action] ** 2 + self.sigma_measure ** 2))

        # Sigma tilde updating for current action
        self.sigma_tilde[state][action] = self.sigma[state][action] / np.sqrt(1 + (self.sigma_measure / self.sigma[state][action]) ** 2)

        # Zeta and KG nu updating for all actions
        max_mu_idx = np.argsort(self.mu[state])[-1]
        another_max_mu_value = np.sort(np.delete(self.mu[state], max_mu_idx))[-1]
        another_max_mu = np.ones([len(self.mu[state])]) * self.mu[state][max_mu_idx]
        another_max_mu[max_mu_idx] = another_max_mu_value

        self.zeta[state] = -1 * np.abs((self.mu[state] - another_max_mu) / self.sigma_tilde[state])

        self.kg_nu[state] = self.sigma_tilde[state] * (self.zeta[state] * norm.cdf(self.zeta[state]) + norm.pdf(self.zeta[state]))

        self.n_episodes += 1
        if done:
            self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)