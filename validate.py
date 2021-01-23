import gym
import uuid
import torch
import minerl
import logging

import numpy as np

from tqdm import tqdm
from tabulate import tabulate

from utils import load_model
from model import ConvNetRGB
from wrappers import FrameSkipWrapper, FrameStackWrapper, GreyScaleWrapper, wrap_env

# logging.basicConfig(level=logging.DEBUG)      


def rollout(env, policy, max_steps=np.inf, video=False, seed=None):
    if video:
        env = wrap_env(env, video=f"videos/{policy.name}")

    if seed is not None:
        env.seed(seed=seed)  # seed for minecraft world generation

    obs, done = env.reset(), False
    total_reward, steps = 0.0, 0.0
    
    while not done and steps <= max_steps:
        action = policy.predict(torch.tensor(obs["pov"]).float())
        
        obs, reward, done, info = env.step(action)
        
        total_reward += reward
        steps += 1

    return total_reward


def validate_policy(policies_with_envs, n_evals=5, **kwargs):
    table = [["policy", f"mean reward (N={n_evals}, args: {kwargs})", "std"]]
    
    for policy, env in tqdm(policies_with_envs):
        policy_rewards = [rollout(env, policy, **kwargs) for _ in range(n_evals)]
        
        table.append([policy.name, np.mean(policy_rewards), np.std(policy_rewards)])

    print(tabulate(table, headers="firstrow", tablefmt="github"))


def main_validate():
    env = gym.make("MineRLTreechop-v0")
    
    policy_env = (
        (load_model("models/stack1_BCE_50_1200_rgb"), wrap_env(env, frame_skip=4)),
        (load_model("models/stack2_BCE_50_1200_rgb"), wrap_env(env, frame_stack=2, frame_skip=4)),
        (load_model("models/stack4_BCE_50_1200_rgb"), wrap_env(env, frame_stack=4, frame_skip=4)),
    )

    with torch.no_grad():
        validate_policy(policy_env, n_evals=15, max_steps=500, seed=23)


def main():
    env = wrap_env(gym.make("MineRLTreechop-v0"), frame_skip=4, frame_stack=4)
    env.make_interactive(port=6666, realtime=True)
    
    model = load_model("models/stack4_BCE_50_1200_rgb")
    
    with torch.no_grad():
        run_reward = rollout(env, model, max_steps=100, video=True)
    

if __name__ == "__main__":
    main()
    # main_validate()