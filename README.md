# RL-Algorithms
# Reinforcement Learning for Autonomous Car Racing

This repo explores **Reinforcement Learning (RL)** techniques to train an autonomous car agent in two simulation environments — **CarRacing-v2 (Gymnasium)** and **CARLA** — using **Stable Baselines3**

---

## Overview

The goal of this task is to train an RL agent to navigate a racetrack autonomously using visual input  
Two environments were tested:

- **CarRacing-v2 (Gymnasium)** — A 96×96 RGB top-down view of a car track with continuous control
- **CARLA Simulator** — A more realistic driving environment with physics and high-fidelity rendering

---

## Environments & Rewards

### **CarRacing-v2 (Gymnasium)**
- **Action Space:** Continuous  
  - `0`: Steering (−1 full left, +1 full right)  
  - `1`: Gas  
  - `2`: Brake  
- **Observation Space:** 96×96×3 RGB image  
- **Rewards:**  
  - `+1000/N` for each track tile visited  
  - `−0.1` every frame (time penalty)  
  - `−100` if the car goes off track  

### **CARLA Simulator**
- A realistic driving environment
- Integration via `modified_run.py` (modified for compatibility with Gym and Stable Baselines3)
