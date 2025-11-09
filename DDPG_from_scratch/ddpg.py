import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import deque
import random
import matplotlib.pyplot as plt
import csv

class ActorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 8, 4),  # (4, 84, 84) -> (16, 20, 20)
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2), # (16, 20, 20) -> (32, 9, 9)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),    # 32*9*9 = 2592
            nn.ReLU(),
            nn.Linear(256, 3),       # 3 continuous actions: steer, gas, brake
            nn.Tanh()                # Output in range [-1, 1]
        )
    
    def forward(self, state):
        return self.net(state)

class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(2592 + 3, 256),  # state features + action
            nn.ReLU(),
            nn.Linear(256, 1)          # single Q-value output
        )
    
    def forward(self, state, action):
        state_features = self.conv(state)
        x = torch.cat([state_features, action], dim=1)
        return self.fc(x)

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resize = cv2.resize(gray, (84, 84))
    return resize / 255.0

memory = deque(maxlen=100000)

actor = ActorNetwork()
critic = CriticNetwork()
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0001)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)

target_actor = ActorNetwork()
target_actor.load_state_dict(actor.state_dict())
target_critic = CriticNetwork()
target_critic.load_state_dict(critic.state_dict())

env = gym.make('CarRacing-v2', render_mode=None, continuous=True) 
gamma = 0.99
tau = 0.001  # soft update parameter

episode_rewards = [] 
total_steps = 0 
normal_scalar = 0.25  # Normal noise scaling

csv_filename = 'ddpg_rewards.csv'
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Episode', 'Reward'])

for episode in range(0, 200):

    frame, info = env.reset()
    frames = deque([preprocess(frame)] * 4, maxlen=4)
    state = np.array(frames)
    total_reward = 0

    for step in range(0, 1000):
        
        with torch.no_grad():
            action = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze().numpy()
        
        # Add noise for exploration
        action = action + np.random.normal(0, normal_scalar, size=action.shape)
        action = np.clip(action, -1, 1)
        
        next_frame, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if step % 100 == 0:
            print(f"Episode {episode}, Step {step}, Total Reward: {total_reward}")
        
        frames.append(preprocess(next_frame))
        next_state = np.array(frames)
        
        memory.append((state, action, reward, next_state, done))
        
        if len(memory) >= 32:
            batch = random.sample(memory, 32)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(np.array(actions), dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)
            
            with torch.no_grad():
                next_actions = target_actor(next_states)
                next_q_values = target_critic(next_states, next_actions).squeeze()
                target_q = rewards + gamma * next_q_values * (1 - dones)
            
            current_q = critic(states, actions).squeeze()
            critic_loss = ((target_q - current_q) ** 2).mean()
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
            predicted_actions = actor(states)
            actor_loss = -critic(states, predicted_actions).mean()  # maximize Q
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            # Soft update target networks
            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data = tau * param.data + (1 - tau) * target_param.data
            
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data = tau * param.data + (1 - tau) * target_param.data

        total_steps += 1
        state = next_state
        
        if done:
            break
    
    episode_rewards.append(total_reward)
    normal_scalar = max(0.05, normal_scalar * 0.995)  # decay exploration noise

    with open(csv_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, total_reward])

    print(f"Episode {episode}/200, Reward: {total_reward}")

    if episode % 100 == 0:
        torch.save(actor.state_dict(), f"ddpg_actor_ep{episode}.pt")
        torch.save(critic.state_dict(), f"ddpg_critic_ep{episode}.pt")
    
torch.save(actor.state_dict(), 'ddpg_actor_final.pt')
torch.save(critic.state_dict(), 'ddpg_critic_final.pt')

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DDPG Training Rewards over Episodes")
plt.legend()
plt.grid(True)
plt.savefig("ddpg_rewards.png")
plt.show()

env.close()
print("Training complete!")