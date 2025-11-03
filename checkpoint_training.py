import gym
import gym_carla
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    params = {
        'number_of_vehicles': 1,
        'number_of_walkers': 0,
        'display_size': 256,
        'max_past_step': 1,
        'dt': 0.1,
        'discrete': True,
        'discrete_acc': [-3.0, 0.0, 3.0],
        'discrete_steer': [-0.2, 0.0, 0.2],
        'continuous_accel_range': [-3.0, 3.0],
        'continuous_steer_range': [-0.3, 0.3],
        'ego_vehicle_filter': 'vehicle.lincoln*',
        'port': 4000,
        'town': 'Town03',
        'max_time_episode': 1000,
        'max_waypt': 12,
        'obs_range': 32,
        'lidar_bin': 0.125,
        'd_behind': 12,
        'out_lane_thres': 2.0,
        'desired_speed': 8,
        'max_ego_spawn_times': 200,
        'display_route': True,
    }

    print("Loading checkpoint!")
    model = DQN.load("./checkpoints/rl_model_2000_steps")
    
    # Create environment
    env = gym.make('carla-v0', params=params)
    model.set_env(env)

    checkpoint = CheckpointCallback(save_freq=2000, save_path='./checkpoints/')
    
    print("Starting training!")
    model.learn(total_timesteps=8000, callback=checkpoint, reset_num_timesteps=False, progress_bar=True)
    
    model.save("carla_rl_final")
    print("Training complete!")
    
    env.close()

if __name__ == '__main__':
    main()