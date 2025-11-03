# RL-Algorithms
# Reinforcement Learning for Autonomous Car Racing

This repo explores **Reinforcement Learning (RL)** techniques to train an autonomous car agent in two simulation environments using using **Stable Baselines3**: **CarRacing-v2 (Gymnasium)** and **CARLA**.

---

## Overview

The goal of this task is to train an RL agent to navigate a racetrack/road system autonomously using visual input  
Two environments:

- **CarRacing-v2 (Gymnasium)** - A 96×96 RGB top-down view of a car track with continuous control
- **CARLA Simulator** - A more realistic driving environment

---

## Environments & Rewards

### **CarRacing-v2 (Gymnasium)**
- **Action Space:** Continuous  
  - 0: Steering (-1 full left, +1 full right)  
  - 1: Gas  
  - 2: Brake  
- **Observation Space:** 96×96×3 RGB image  
- **Rewards:**  
  - +1000/N for each track tile visited  
  - -0.1 every frame (time penalty)  
  - -100 if the car goes off track
  
### **CARLA Simulator**
- A realistic driving environment
- Using modified_run.py

## CarRacing-v2 PPO (100k Steps)
[![Watch the video](https://img.youtube.com/vi/IEGeH7PTIR0/hqdefault.jpg)](https://youtu.be/IEGeH7PTIR0)

## CarRacing-v2 PPO (500k Steps)
[![Watch the video](https://img.youtube.com/vi/xIkj2q2np7k/hqdefault.jpg)](https://youtu.be/xIkj2q2np7k)
