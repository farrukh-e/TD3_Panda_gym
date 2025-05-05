import gymnasium as gym
import numpy as np
# from TD3 import TD3, ReplayBuffer   # make sure your TD3 code is importable
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

# --------------------------- configuration ---------------------------
ENV_NAME        = "Pendulum-v1"
EPISODES        = 50          # training episodes
MAX_STEPS       = 200          # env horizon
BATCH_SIZE      = 256
BUFFER_CAPACITY = 100_000
EXPL_NOISE_STD  = 0.1          # σ for Gaussian exploration
SAVE_PATH       = "td3_pendulum"  # prefix for .h5 weight files
# --------------------------------------------------------------------

def train():
    env = gym.make(ENV_NAME)
    obs_dim  = env.observation_space.shape[0]
    act_dim  = env.action_space.shape[0]
    max_act  = float(env.action_space.high[0])
    
    # Create action noise for exploration
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=EXPL_NOISE_STD * max_act * np.ones(n_actions)
    )
    
    # Initialize the TD3 agent
    agent = TD3(
        policy="MlpPolicy",
        env=env,
        action_noise=action_noise,
        buffer_size=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # Train the agent
    total_timesteps = EPISODES * MAX_STEPS
    agent.learn(total_timesteps=total_timesteps)
    
    # Save the agent
#    agent.save(SAVE_PATH)
    env.close()
    return agent

'''
# Original implementation commented out
def train():
    env = gym.make(ENV_NAME)
    obs_dim  = env.observation_space.shape[0]
    act_dim  = env.action_space.shape[0]
    max_act  = float(env.action_space.high[0])

    agent  = TD3(obs_dim, act_dim, max_act)
    buffer = ReplayBuffer(BUFFER_CAPACITY, input_shape=(obs_dim,), n_actions=act_dim)

    for ep in range(EPISODES):
        obs, _ = env.reset()
        done, ep_ret, step = False, 0.0, 0
        while not done and step < MAX_STEPS:
            # policy action + Gaussian exploration noise (clipped)
            action = agent.select_action(obs)
            action = np.clip(
                action + np.random.normal(0, EXPL_NOISE_STD * max_act, size=act_dim),
                -max_act, max_act
            )

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, reward, next_obs, float(done))

            # begin updating as soon as the buffer can yield a full batch
            if buffer.size >= BATCH_SIZE:
                agent.train(buffer, BATCH_SIZE)

            obs, ep_ret, step = next_obs, ep_ret + reward, step + 1

        print(f"Episode {ep+1:3d} | return = {ep_ret:8.2f}")
        if (ep + 1) % 50 == 0:
            agent.save(SAVE_PATH)

    agent.save(SAVE_PATH)      # final checkpoint
    env.close()
    return agent
'''

def evaluate(agent, episodes=10, render=False):
    env = gym.make(ENV_NAME, render_mode="human" if render else None)
    
    # Use Stable Baselines evaluation function
    mean_reward, std_reward = evaluate_policy(
        agent, 
        env, 
        n_eval_episodes=episodes, 
        deterministic=True
    )
    
    print(f"Average return over {episodes} eval episodes: {mean_reward:.2f} ± {std_reward:.2f}")
    env.close()
    return mean_reward

'''
# Original evaluation function commented out
def evaluate(agent, episodes=10, render=False):
    env = gym.make(ENV_NAME)
    avg_ret = 0.0
    for ep in range(episodes):
        obs, _ = env.reset()
        done, ep_ret, step = False, 0.0, 0

        while not done and step < MAX_STEPS:
            action = agent.select_action(obs)      # ***no exploration noise***
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward
            step   += 1
            if render:
                env.render()
        print(f"[eval] episode {ep+1} | return = {ep_ret:8.2f}")
        avg_ret += ep_ret
    env.close()
    print(f"Average return over {episodes} eval episodes: {avg_ret / episodes:.2f}")
'''

if __name__ == "__main__":
    trained_agent = train()
    evaluate(trained_agent, episodes=10, render=False)
