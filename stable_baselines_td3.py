import gymnasium as gym
import numpy as np
import panda_gym
from stable_baselines3 import HerReplayBuffer, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.evaluation import evaluate_policy

ENV_NAME        = "PandaReach-v3"
#Parameters are based on original TD3 paper
EPISODES        = 20
MAX_STEPS       = 200
BATCH_SIZE      = 256
BUFFER_CAPACITY = 100_000
EXPL_NOISE_STD  = 0.1
TAU             = 0.005
DISCOUNT        = 0.99
POLICY_FREQ    = 2
POLICY_NOISE    = 0.2
NOISE_CLIP     = 0.5
SAVE_PATH       = "td3_her_reach"

def train():
    env = gym.make(ENV_NAME, render_mode="human")
    
    # Set up the action noise for exploration
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=EXPL_NOISE_STD * np.ones(n_actions)
    )
    
    model = TD3(
        "MultiInputPolicy", 
        env,
        replay_buffer_class=HerReplayBuffer,
        # HER specific parameters
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy=GoalSelectionStrategy.FUTURE,
            # Remove both problematic parameters
        ),
        action_noise=action_noise,
        buffer_size=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=DISCOUNT,
        policy_delay=POLICY_FREQ,
        target_policy_noise=POLICY_NOISE,
        target_noise_clip=NOISE_CLIP,
        learning_rate=1e-3,
        verbose=1
    )
    
    total_timesteps = EPISODES * MAX_STEPS
    model.learn(total_timesteps=total_timesteps)
    
    # Save the trained model
#    model.save(SAVE_PATH)
    env.close()
    return model

def evaluate(agent, episodes=10, render=False):
    # Create environment for evaluation
    env = gym.make(ENV_NAME, render_mode="human" if render else None)
    
    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(
        agent, 
        env, 
        n_eval_episodes=episodes,
        deterministic=True
    )
    
    print(f"Average return over {episodes} eval episodes: {mean_reward:.2f} ± {std_reward:.2f}")
    env.close()
    return mean_reward

if __name__ == "__main__":
    trained_agent = train()
    evaluate(trained_agent, episodes=10, render=False)
