import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("ppo_car_racing_500k")

env = gym.make('CarRacing-v2', render_mode='human', continuous=True)

obs, info = env.reset()
total_reward = 0
episode_over = False
    
while not episode_over: 
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
        
    episode_over = terminated or truncated

print(f"Episode finished! Total Reward: {total_reward}")

env.close()