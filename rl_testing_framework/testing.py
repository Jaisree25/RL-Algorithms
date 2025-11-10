import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import deque
import random
import matplotlib.pyplot as plt

# Networks
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )
    
    def forward(self, state):
        return self.net(state)

class ActorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Tanh()
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
            nn.Linear(2592 + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        state_features = self.conv(state)
        x = torch.cat([state_features, action], dim=1)
        return self.fc(x)

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resize = cv2.resize(gray, (84, 84))
    return resize / 255.0

# DQN Training
def train_dqn(n_episodes=100):
    print("Training DQN")
    
    memory = deque(maxlen=100000)
    q_network = QNetwork()
    target_network = QNetwork()
    target_network.load_state_dict(q_network.state_dict())
    optimizer = torch.optim.Adam(q_network.parameters(), lr=0.0001)
    
    env = gym.make('CarRacing-v2', render_mode=None, continuous=False)
    epsilon = 1.0
    gamma = 0.99
    episode_rewards = []
    episode_losses = []
    epsilon_values = []
    total_steps = 0
    
    for episode in range(n_episodes):
        frame, _ = env.reset()
        frames = deque([preprocess(frame)] * 4, maxlen=4)
        state = np.array(frames)
        total_reward = 0
        losses = []
        
        for step in range(1000):
            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                with torch.no_grad():
                    action = q_network(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()
            
            next_frame, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
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
                    next_q = target_network(next_states).max(1)[0]
                    target_q = rewards + gamma * next_q * (1 - dones)
                
                current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
                loss = ((target_q - current_q) ** 2).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            total_steps += 1
            if total_steps % 1000 == 0:
                target_network.load_state_dict(q_network.state_dict())
            
            state = next_state
            if done:
                break
        
        epsilon = max(0.1, epsilon * 0.995)

        episode_rewards.append(total_reward)

        if losses:
            episode_losses.append(np.mean(losses))
        else:
            episode_losses.append(0)

        epsilon_values.append(epsilon)
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{n_episodes} | Reward: {total_reward} | Loss: {episode_losses[-1]}")
    
    env.close()
    torch.save(q_network.state_dict(), 'dqn_final.pt')
    return episode_rewards, episode_losses, epsilon_values

# DDPG Training
def train_ddpg(n_episodes=100):
    print("Training DDPG")
    
    memory = deque(maxlen=100000)
    actor = ActorNetwork()
    critic = CriticNetwork()
    target_actor = ActorNetwork()
    target_critic = CriticNetwork()
    
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0001)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0001)
    
    env = gym.make('CarRacing-v2', render_mode=None, continuous=True)
    gamma = 0.99
    tau = 0.001
    normal_scalar = 0.25
    episode_rewards = []
    episode_losses = []
    noise_values = []
    
    for episode in range(n_episodes):
        frame, _ = env.reset()
        frames = deque([preprocess(frame)] * 4, maxlen=4)
        state = np.array(frames)
        total_reward = 0
        losses = []
        
        for step in range(1000):
            with torch.no_grad():
                action = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze().numpy()
            
            action = action + np.random.normal(0, normal_scalar, size=action.shape)
            action = np.clip(action, -1, 1)
            
            next_frame, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
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
                    next_q = target_critic(next_states, next_actions).squeeze()
                    target_q = rewards + gamma * next_q * (1 - dones)
                
                current_q = critic(states, actions).squeeze()
                critic_loss = ((target_q - current_q) ** 2).mean()
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                predicted_actions = actor(states)
                actor_loss = -critic(states, predicted_actions).mean()
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data = tau * param.data + (1 - tau) * target_param.data
                
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data = tau * param.data + (1 - tau) * target_param.data
                
                losses.append(critic_loss.item())
            
            state = next_state
            if done:
                break
        
        normal_scalar = max(0.05, normal_scalar * 0.995)

        episode_rewards.append(total_reward)

        if losses:
            episode_losses.append(np.mean(losses))
        else:
            episode_losses.append(0)

        noise_values.append(normal_scalar)
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{n_episodes} | Reward: {total_reward} | Loss: {episode_losses[-1]}")
    
    env.close()
    torch.save(actor.state_dict(), 'ddpg_actor_final.pt')
    torch.save(critic.state_dict(), 'ddpg_critic_final.pt')
    return episode_rewards, episode_losses, noise_values

# Plotting
def save_plots(dqn_data, ddpg_data):
    dqn_rewards, dqn_losses, dqn_epsilon = dqn_data
    ddpg_rewards, ddpg_losses, ddpg_noise = ddpg_data
    
    # Plot 1: Rewards
    plt.figure(figsize=(10, 5))
    plt.plot(dqn_rewards, label='DQN')
    plt.plot(ddpg_rewards, label='DDPG')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards')
    plt.legend()
    plt.grid(True)
    plt.savefig('rewards.png', dpi=300)
    plt.close()
    print("Saved rewards.png")
    
    # Plot 2: Loss
    plt.figure(figsize=(10, 5))
    plt.plot(dqn_losses, label='DQN')
    plt.plot(ddpg_losses, label='DDPG')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.png', dpi=300)
    plt.close()
    print("Saved loss.png")
    
    # Plot 3: Exploration (Epsilon/Noise)
    plt.figure(figsize=(10, 5))
    plt.plot(dqn_epsilon, label='DQN (Epsilon)')
    plt.plot(ddpg_noise, label='DDPG (Noise)')
    plt.xlabel('Episode')
    plt.ylabel('Exploration Rate')
    plt.title('Exploration Decay')
    plt.legend()
    plt.grid(True)
    plt.savefig('exploration.png', dpi=300)
    plt.close()
    print("Saved exploration.png")

print("RL Benchmark")
    
n_episodes = 100
    
dqn_data = train_dqn(n_episodes)
ddpg_data = train_ddpg(n_episodes)
    
save_plots(dqn_data, ddpg_data)
    
print("RL Benchmark Complete!")