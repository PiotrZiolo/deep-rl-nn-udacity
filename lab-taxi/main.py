from agent import Agent, Sarsa, QLearning, ExpectedSarsa, QLearningGuided, KG
from monitor import interact
import gym
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3').env  # If you use gym.make('Taxi-v3'), then a limit of 200 steps is applied

game_map = np.array(["".join((c.decode("utf-8") for c in line[0:])) for line in env.desc.tolist()]).reshape((-1, 1))
print(game_map)

sns.set(rc={'figure.figsize': (11.7, 8.27)})

# agent_names = ["Sarsa", "Q-Learning", "Expected Sarsa", "Guided Q-Learning", "KG"]
# agent_classes = [Sarsa, QLearning, ExpectedSarsa, QLearningGuided, KG]

agent_names = ["KG"]
agent_classes = [KG]

only_KG = True

data = pd.DataFrame([], columns=['Agent', 'episode', 'reward'])
if only_KG:
    data_agent = pd.DataFrame([], columns=['Agent', 'episode', 'greedy', 'mu', 'nu'])

n_episodes = 100000

for i in range(len(agent_classes)):
    agent = agent_classes[i](env)
    agent_name = agent_names[i]

    avg_rewards, best_avg_reward = interact(env, agent, num_episodes=n_episodes, window=n_episodes)

    data_new = pd.DataFrame(list(avg_rewards), columns=['reward'])
    data_new.loc[:, 'episode'] = range(0, len(list(avg_rewards)))
    data_new.loc[:, 'Agent'] = agent_name
    data = data.append(data_new)

    if only_KG:
        data_agent_new = pd.DataFrame(agent.greedy_choice, columns=['greedy'])
        data_agent_new.loc[:, 'episode'] = range(0, len(agent.greedy_choice))
        data_agent_new.loc[:, 'Agent'] = agent_name
        mu = [x[0] for x in agent.mu_vs_nu]
        nu = [x[1] for x in agent.mu_vs_nu]
        data_agent_new.loc[:, 'mu'] = mu
        data_agent_new.loc[:, 'nu'] = nu
        data_agent = data_agent.append(data_agent_new)

if only_KG:
    sns_plot = sns.lineplot(x='episode', y='greedy', hue='Agent', data=data_agent)
    sns_plot.set(ylim=(-0.2, 1.2))
    # sns_plot.figure.savefig("tax1v3.png")
    plt.show()

    data_mu_nu = pd.DataFrame([], columns=['var'])
    data_mu_nu.loc[:, 'var'] = ['mu']*len(data_agent['mu']) + ['nu']*len(data_agent['nu'])
    data_mu_nu.loc[:, 'reward estimate'] = data_agent['mu'].tolist() + data_agent['nu'].tolist()
    data_mu_nu.loc[:, 'episode'] = list(range(0, len(data_agent['mu']))) + list(range(0, len(data_agent['nu'])))
    sns_plot = sns.lineplot(x='episode', y='reward estimate', hue='var', data=data_mu_nu)
    mu_min = np.min(data_agent['mu'])
    mu_max = np.max(data_agent['mu'])
    mu_range = mu_max - mu_min
    sns_plot.set(ylim=(mu_min - 5 * mu_range, mu_max + 5 * mu_range))
    # sns_plot.figure.savefig("tax1v3.png")
    plt.show()

sns_plot = sns.lineplot(x='episode', y='reward', hue='Agent', data=data)
sns_plot.set(ylim=(-500, 20))
# sns_plot.figure.savefig("tax1v3.png")
plt.show()

sns_plot = sns.lineplot(x='episode', y='reward', hue='Agent', data=data)
sns_plot.set(ylim=(-60, 20))
# sns_plot.figure.savefig("tax1v3.png")
plt.show()

# sns.set(rc={'figure.figsize': (11.7, 8.27)})
# sns_plot = sns.lineplot(x='episode', y='reward', data=data.iloc[-1000:])
# # sns_plot.figure.savefig("tax1v3.png")
# plt.show()
#
# for i in range(10):
#     state = env.reset()
#     env.render()
#     # initialize the sampled reward
#
#     samp_reward = 0
#     while True:
#         # agent selects an action
#         action = agent.select_action(state)
#         # agent performs the selected action
#         next_state, reward, done, _ = env.step(action)
#         env.render()
#         # agent performs internal updates based on sampled experience
#         agent.step(state, action, reward, next_state, done)
#         # update the sampled reward
#         samp_reward += reward
#         print(samp_reward)
#         # update the state (s <- s') to next time step
#         state = next_state
#         if done:
#             break

