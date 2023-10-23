import csv
import logs
import numpy as np
import pandas as pd
import seaborn as sns
from Bandit import Bandit
import matplotlib.pyplot as plt

Bandit_Reward = [1, 2, 3, 4]
Trials = 20000

class EpsilonGreedy(Bandit):
    def __init__(self, probabilities, epsilon=0.1):
        self.epsilon = epsilon
        self.N = len(probabilities)
        self.k = np.zeros(self.N)
        self.mean_reward = np.zeros(self.N)
        self.total_reward = 0
        self.rewards = []

    def __repr__(self):
        return f"EpsilonGreedy(epsilon={self.epsilon}, N={self.N}, mean_reward={self.mean_reward})"

    def pull(self):
        if np.random.rand() < self.epsilon:
            selected_bandit = np.random.choice(self.N)
        else:
            selected_bandit = np.argmax(self.mean_reward)
        reward = Bandit_Reward[selected_bandit]
        self.rewards.append((selected_bandit, reward, "EpsilonGreedy"))
        return selected_bandit, reward

    def update(self, selected_bandit, reward):
        self.k[selected_bandit] += 1
        alpha = 1.0 / self.k[selected_bandit]
        self.mean_reward[selected_bandit] += alpha * (reward - self.mean_reward[selected_bandit])
        self.total_reward += reward

    def experiment(self, trials=Trials):
        for t in range(1, trials + 1):
            selected_bandit, reward = self.pull()
            self.update(selected_bandit, reward)
            self.epsilon = max(self.epsilon / t, 0.01)

    def report(self):
        with open("rewards.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Bandit", "Reward", "Algorithm"])
            writer.writerows(self.rewards)
        
        cumulative_reward = np.sum([reward for _, reward, _ in self.rewards])
        print(f"Epsilon-Greedy Total Reward: {cumulative_reward}")
        
        optimal_reward = max(Bandit_Reward)
        cumulative_regret = optimal_reward * len(self.rewards) - cumulative_reward
        print(f"Epsilon-Greedy Total Regret: {cumulative_regret}")
        
        avg_reward = self.total_reward / sum(self.k)
        avg_regret = optimal_reward - avg_reward
        print(f"Mean Reward: {avg_reward:.2f}")
        print(f"Mean Regret: {avg_regret:.2f}")

class ThompsonSampling(Bandit):
    def __init__(self, probabilities):
        self.N = len(probabilities)
        self.alpha = np.ones(self.N)
        self.beta = np.ones(self.N)
        self.total_reward = 0
        self.rewards = []

    def __repr__(self):
        return f"ThompsonSampling(N={self.N}, alpha={self.alpha}, beta={self.beta})"
    
    def pull(self):
        sampled_values = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.N)]
        selected_bandit = np.argmax(sampled_values)
        reward = Bandit_Reward[selected_bandit]
        self.rewards.append((selected_bandit, reward, "ThompsonSampling"))
        return selected_bandit, reward

    def update(self, selected_bandit, reward):
        if reward > 0:
            self.alpha[selected_bandit] += 1
        else:
            self.beta[selected_bandit] += 1
        self.total_reward += reward

    def experiment(self, trials=Trials):
        for _ in range(trials):
            selected_bandit, reward = self.pull()
            self.update(selected_bandit, reward)

    def report(self):
        with open("rewards.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.rewards)
        
        cumulative_reward = np.sum([reward for _, reward, _ in self.rewards])
        print(f"Thompson Sampling Total Reward: {cumulative_reward}")
        
        optimal_reward = max(Bandit_Reward)
        cumulative_regret = optimal_reward * len(self.rewards) - cumulative_reward
        print(f"Thompson Sampling Total Regret: {cumulative_regret}")
        
        avg_reward = self.total_reward / (sum(self.alpha) + sum(self.beta) - 2 * self.N)
        avg_regret = optimal_reward - avg_reward
        print(f"Mean Reward: {avg_reward:.2f}")
        print(f"Mean Regret: {avg_regret:.2f}")

class Visualization():

    def plot1(self, eg_rewards, ts_rewards):
        eg_cumulative_rewards = np.cumsum([reward for _, reward, _ in eg_rewards])
        ts_cumulative_rewards = np.cumsum([reward for _, reward, _ in ts_rewards])
        
        eg_avg_rewards = eg_cumulative_rewards / (np.arange(len(eg_rewards)) + 1)
        ts_avg_rewards = ts_cumulative_rewards / (np.arange(len(ts_rewards)) + 1)
        
        plt.figure(figsize=(12, 6))
        plt.plot(eg_avg_rewards, label="Epsilon-Greedy")
        plt.plot(ts_avg_rewards, label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Mean Reward")
        plt.title("Learning Curve of E-Greedy and Thompson Sampling")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot2(self, eg_rewards, ts_rewards):
        eg_cumulative_rewards = np.cumsum([reward for _, reward, _ in eg_rewards])
        ts_cumulative_rewards = np.cumsum([reward for _, reward, _ in ts_rewards])
        
        plt.figure(figsize=(12, 6))
        plt.plot(eg_cumulative_rewards, label="Epsilon-Greedy")
        plt.plot(ts_cumulative_rewards, label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Total Reward")
        plt.title("Total Rewards of E-Greedy and Thompson Sampling")
        plt.legend()
        plt.grid(True)
        plt.show()

epsilon_greedy = EpsilonGreedy(Bandit_Reward)
thompson_sampling = ThompsonSampling(Bandit_Reward)

epsilon_greedy.experiment()
thompson_sampling.experiment()

epsilon_greedy.report()
thompson_sampling.report()

viz = Visualization()
viz.plot1(epsilon_greedy.rewards, thompson_sampling.rewards)
viz.plot2(epsilon_greedy.rewards, thompson_sampling.rewards)