import numpy as np
import pandas as pd
import gymnasium as gym  # Use Gymnasium instead of Gym
from gymnasium import spaces
from stable_baselines3 import PPO
import torch.nn as nn
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Define a custom feature extractor
class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomMLP, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 8192*2),
            nn.ReLU(),
            nn.Linear(8192*2, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.mlp(observations)


# Custom Policy with Feature Extractor
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           features_extractor_class=CustomMLP,  # Use custom MLP
                                           features_extractor_kwargs={"features_dim": 128}
                                           )


# Define Custom Trading Environment
class SP500TradingEnv(gym.Env):
    def __init__(self, df, window_size=60_000):
        super(SP500TradingEnv, self).__init__()

        self.df = df
        self.window_size = window_size
        self.current_step = self.window_size
        self.done = False

        self.action_space = spaces.Discrete(3)  # (0: idle, 1: long, 2: short)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size + 1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Ensures compatibility with Gym's API
        self.current_step = self.window_size
        self.done = False
        return self._get_observation(), {}

    def _get_observation(self):
        return np.append(
            self.df["Price"].iloc[self.current_step - self.window_size:self.current_step].values,
            self.df["Active_Hours"].iloc[self.current_step]
        )

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, self.done, False, {}

        start_price = self.df["Price"].iloc[self.current_step]
        future_prices = self.df["Price"].iloc[self.current_step:self.current_step + 14_400]

        reward = 0
        new_step = self.current_step + 1  # Default advance 1 min (idle)

        if action == 1:  # Long
            for i, price in enumerate(future_prices):
                if (price - start_price) / start_price >= 0.01:
                    reward = 1
                    new_step = self.current_step + i
                    break
                elif (price - start_price) / start_price <= -0.02:
                    reward = -5
                    new_step = self.current_step + i
                    break

        elif action == 2:  # Short
            for i, price in enumerate(future_prices):
                if (start_price - price) / start_price >= 0.01:
                    reward = 1
                    new_step = self.current_step + i
                    break
                elif (start_price - price) / start_price <= -0.02:
                    reward = -5
                    new_step = self.current_step + i
                    break

        if action in [1, 2] and reward == 0:
            reward = 0  # Invalid trade penalty

        self.current_step = new_step
        self.done = self.current_step >= len(self.df) - 14_400

        return self._get_observation(), reward, self.done, False, {}


# Load and preprocess SP500 data
df = pd.read_csv("sp500_data_trimmed.csv", delimiter=";", header=None, names=["Date", "Time", "Price", "H", "L", "C", "Vol"])

# Keep only Time and Price
df = df[["Time", "Price"]]
# Convert time to datetime format
df["Time"] = pd.to_datetime(df["Time"], format="%H:%M").dt.time

# Define active market hours (9:30 AM - 4:00 PM EST)
def is_active_market(time):
    return time >= pd.to_datetime("09:30", format="%H:%M").time() and \
           time <= pd.to_datetime("16:00", format="%H:%M").time()

df["Active_Hours"] = df["Time"].apply(is_active_market).astype(int)

# Save processed data for training
df.to_csv("processed_sp500_data.csv", index=False)

print(df.head())

# Create Trading Environment
env = DummyVecEnv([lambda: SP500TradingEnv(df)])

# Train PPO Model with Custom MLP
log_dir = "./ppo_sp500_logs"
model = PPO(CustomPolicy, env,
            learning_rate=0.0001,  # Increase learning rate
            n_steps=4096,          # Collect more data per update
            batch_size=64,        # Use a larger batch size
            ent_coef=0.01,         # Encourage more exploration
            verbose=1, tensorboard_log=log_dir, device='auto')
model.learn(total_timesteps=3_000_000)

# Save and Load Model
model.save("sp500_trading_model")
model = PPO.load("sp500_trading_model")

# Test the trained model
env = SP500TradingEnv(df)
obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    if done:
        break
