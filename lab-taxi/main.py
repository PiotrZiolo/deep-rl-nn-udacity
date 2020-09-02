from agent import Agent, Sarsa, QLearning, ExpectedSarsa
from monitor import interact
import gym
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# env = gym.make('Taxi-v3')
# agent = Sarsa(env.nA)
# avg_rewards, best_avg_reward = interact(env, agent)
# print("avg_rewards={} best_avg_reward={}".format(avg_rewards, best_avg_reward))
#
# env = gym.make('Taxi-v3')
# agent = QLearning(env.nA)
# avg_rewards, best_avg_reward = interact(env, agent)
# print("avg_rewards={} best_avg_reward={}".format(avg_rewards, best_avg_reward))

env = gym.make('Taxi-v3').env  # If you use gym.make('Taxi-v3'), then a limit of 200 steps is applied
agent = ExpectedSarsa(env.nA)

game_map = np.array(["".join((c.decode("utf-8") for c in line[0:])) for line in env.desc.tolist()]).reshape((-1, 1))
print(game_map)
avg_rewards, best_avg_reward = interact(env, agent)
print("avg_rewards={} best_avg_reward={}".format(avg_rewards, best_avg_reward))

data = pd.DataFrame(list(avg_rewards), columns=['reward'])
data.loc[:, 'episode'] = range(0, len(list(avg_rewards)))

sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns_plot = sns.lineplot(x='episode', y='reward', data=data)
# sns_plot.figure.savefig("tax1v3.png")
plt.show()

sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns_plot = sns.lineplot(x='episode', y='reward', data=data.iloc[-1000:])
# sns_plot.figure.savefig("tax1v3.png")
plt.show()

for i in range(10):
    state = env.reset()
    env.render()
    # initialize the sampled reward

    samp_reward = 0
    while True:
        # agent selects an action
        action = agent.select_action(state)
        # agent performs the selected action
        next_state, reward, done, _ = env.step(action)
        env.render()
        # agent performs internal updates based on sampled experience
        agent.step(state, action, reward, next_state, done)
        # update the sampled reward
        samp_reward += reward
        print(samp_reward)
        # update the state (s <- s') to next time step
        state = next_state
        if done:
            break

