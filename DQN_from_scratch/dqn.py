import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import deque
import random
import matplotlib.pyplot as plt

class QNetwork(nn.Module):
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
            nn.Linear(256, 5)        # 5 discrete actions
        )
    
    def forward(self, state):
        return self.net(state)

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resize = cv2.resize(gray, (84, 84))
    return resize / 255.0

memory = deque(maxlen=100000)

q_network = QNetwork()
optimizer = torch.optim.Adam(q_network.parameters(), lr=0.0001)
env = gym.make('CarRacing-v2', render_mode=None, continuous=False)
epsilon = 1
gamma = 0.99

episode_rewards = [] 

for episode in range(0, 200):

    frame, info = env.reset()
    frames = deque([preprocess(frame)] * 4, maxlen=4)
    state = np.array(frames)
    total_reward = 0

    for step in range(0, 1000):
        
        if random.random() < epsilon:
            action = random.randint(0, 4)
        else:
            action = q_network(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()
        
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
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)
            
            with torch.no_grad():
                next_q_values = q_network(next_states).max(1)[0]
                target_q = rewards + gamma * next_q_values * (1 - dones)
            
            current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
            
            loss = ((target_q - current_q) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        state = next_state
        
        if done:
            break
    
    episode_rewards.append(total_reward)
    epsilon = max(0.1, epsilon * 0.995)
    
    print(f"Episode {episode}/200, Reward: {total_reward}")

    if episode % 100 == 0:
        torch.save(q_network.state_dict(), f"DQN_ep{episode}.pt")
    
torch.save(q_network.state_dict(), 'dqn.pt')

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Rewards over Episodes")
plt.legend()
plt.grid(True)
plt.savefig("rewards.png")
plt.show()

env.close()
print("Training complete!")