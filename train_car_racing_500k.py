import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('CarRacing-v2', render_mode='rgb_array', continuous=True)

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./car_racing_tensorboard/")

print("Training started!")
model.learn(total_timesteps=500000, progress_bar=True)

model.save("ppo_car_racing_500k")
print("Training complete!")

env.close()