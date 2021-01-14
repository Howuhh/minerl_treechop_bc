import uuid
import gym
import torch
import minerl
import logging
import imageio

import numpy as np

from tabulate import tabulate

from model import ConvNetRGB
from wrappers import FrameSkipWrapper, FrameStackWrapper, GreyScaleWrapper
from utils import load_model

logging.basicConfig(level=logging.DEBUG)
        
        
def rollout(env, policy, max_steps=50, video=False):
    video_frames = []
    
    obs, done = env.reset(), False
    total_reward, steps = 0.0, 0.0
    
    while not done and steps <= max_steps:
        if video:
            video_frames.append(obs["pov"][:, :, :3])

        action = policy.predict(torch.tensor(obs["pov"]).float())
        obs, reward, done, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
    if video:
        imageio.mimwrite(f"videos/{policy.name}_rollout{str(uuid.uuid4())}.mp4", video_frames, fps=30.0)
        
    return total_reward


# TODO: для каждой модели нужен свой враппер для env, добавить
def validate_policy(env, policies, **kwargs):
    if not isinstance(policies, list):
        policies = [policies]
    
    table = [["policy", f"mean reward (N=5, args: {kwargs})", "std"]]
    
    for policy in policies:
        policy_rewards = [rollout(env, policy, **kwargs) for _ in range(5)]
        
        policy_name = getattr(policy, "name", policy.__class__.__name__)
        table.append([policy_name, np.mean(policy_rewards), np.std(policy_rewards)])

    print(tabulate(table, headers="firstrow", tablefmt="github"))
        

def main():
    env = FrameSkipWrapper(
            FrameStackWrapper(gym.make("MineRLTreechop-v0"), 4)
        )
    env.make_interactive(port=6666, realtime=True)
    
    model = load_model("models/model_stack4_BCE_50_1200")
    
    with torch.no_grad():
        run_reward = rollout(env, model, max_steps=50)
    

if __name__ == "__main__":
    main()