from math import floor
import numpy as np

a = np.array([1, 2, 3, 4, 3, 5])
b = [0, 3]

print(a[b])


class QLearningGuided(object):

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

    def choose_action(self, state, action_list):
        nA = len(action_list)
        v = self.Q[state]
        if self.epsilon > 0.000001:
            action = min(floor(np.random.rand() / (self.epsilon / nA)), nA)
            if action == nA:  # select greedy action
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