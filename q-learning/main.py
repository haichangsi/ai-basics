import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import matplotlib.pyplot as plt

map = generate_random_map(size=4, p=0.4)

hard_6x6_map = [
    "SFFHFF",
    "FHFFHF",
    "FFFHFF",
    "HFFFFH",
    "FFHFFF",
    "FHFFFG"
]

env = gym.make('FrozenLake-v1', map_name="4x4", desc=map, is_slippery=False)
states = env.observation_space.n
actions = env.action_space.n
q_table = np.zeros((states, actions))

alpha = 0.8
gamma = 0.8
epsilon = 1.0
epsilon_min = 0.001
epsilon_decay = 0.995

observation, info = env.reset()

all_count = 0
rewards = []

def q_get_action(observation):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return argmax(q_table[observation, :])

def eps_greedy_policy(observation):
    return q_get_action(observation)

def argmax(arr):
    arr_max = np.max(arr)
    return np.random.choice(np.where(arr == arr_max)[0])

for i in range(1000):
    observation, info = env.reset()
    done = False
    rewards_sum = 0
    while not done:
        action = eps_greedy_policy(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        q_table[observation, action] = (1 - alpha) * q_table[observation, action] + alpha * (reward + gamma * np.max(q_table[next_observation, :]))
        observation = next_observation
        rewards_sum += reward
        all_count += 1
        done = terminated or truncated

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(rewards_sum)

env.close()

window_size = 100
rewards_smooth = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(12, 6))
plt.plot(rewards_smooth, label='Smoothed Rewards per Episode')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Rewards per Episode in FrozenLake (Smoothed)')
plt.legend()
plt.savefig('seed98_alpha08.png')
plt.show()

# print(rewards[-10:])
# print(rewards_smooth[-10:])

env_human = gym.make('FrozenLake-v1', map_name="4x4", desc=map, is_slippery=False, render_mode='human')
observation, info = env_human.reset()
for i in range(1):
    done = False
    while not done:
        action = eps_greedy_policy(observation)
        observation, reward, terminated, truncated, info = env_human.step(action)
        env_human.render()
        if terminated or truncated:
            observation, info = env_human.reset()
            done = True

env_human.close()
