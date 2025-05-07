import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
import panda_gym

ENV_NAME = "PandaReach-v3"
SEED      = 42
DEMO_STEPS = 10_000      # how many expert transitions to copy
TOTAL_STEPS = 10000     # TD3 training horizon
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

def make_env(gui=False):
    env = gym.make(
        ENV_NAME,
        reward_type="dense",
        control_type="ee",
        render_mode="human" if gui else "rgb_array",
    )
    env.reset(seed=SEED)
    return env     

set_random_seed(SEED)
train_env = make_env(gui=False)   # DIRECT → no GUI limit
eval_env  = make_env(gui=False)   # another DIRECT client, safe
demo_env  = make_env(gui=True)    # only one GUI connection

# load trained PPO expert
custom_objects = {
    # replace both schedules with flat constants
    "clip_range": 0.2,
    "lr_schedule": 3e-4,
}

expert = PPO.load(
    "PandaReach_PPO_v3_ee_model.zip",
    device="cpu",           # or "cuda" if you like
    custom_objects=custom_objects,
)

n_actions = train_env.action_space.shape[0]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=EXPL_NOISE_STD * np.ones(n_actions)
)

model = TD3(
    "MultiInputPolicy", 
    train_env,
    action_noise=action_noise,
    buffer_size=BUFFER_CAPACITY,
    batch_size=BATCH_SIZE,
    tau=TAU,
    gamma=DISCOUNT,
    policy_delay=POLICY_FREQ,
    target_policy_noise=POLICY_NOISE,
    target_noise_clip=NOISE_CLIP,
    learning_rate=1e-3,
    seed=SEED,
    verbose=1
)

# ---- 1) pre‑fill with expert data -----------------------------------------
obs, _ = train_env.reset()
for _ in range(DEMO_STEPS):
    act, _     = expert.predict(obs, deterministic=True)
    next_obs, rew, term, trunc, infos = train_env.step(act)
    done       = term or trunc
    model.replay_buffer.add(obs, next_obs, act, rew, done, [infos])
    obs        = next_obs if not done else train_env.reset()[0]

model.learning_starts = model.replay_buffer.size()
# ---------------------------------------------------------------------------

eval_cb = EvalCallback(eval_env, eval_freq=2_000, deterministic=True)
model.learn(TOTAL_STEPS, callback=eval_cb)
model.save("td3_panda_imitation")

print("Mean reward after training:",
      eval_cb.last_mean_reward)
