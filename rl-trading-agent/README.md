# Deep Q-Learning Trading Agent
Reinforcement Learning agent (DQN) trained for automated trading decision-making.


**Note:** agent.py now includes a complete DQN training loop (replay buffer, target network, Bellman update) plus a tiny synthetic ToyTradingEnv so it runs end to end with zero external data. The previous version only had the action-selection path with no learning step at all. Swap ToyTradingEnv for a real market-data gymnasium environment before trusting this for anything beyond a teaching example - DQN on raw price deltas like this is not a trading strategy.
