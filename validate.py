import uuid
import gym
import torch
import minerl
import logging
import imageio

import numpy as np

from tabulate import tabulate

from model import ConvNetRGB
from wrappers import FrameSkipWrapper, FrameStackWrapper
from utils import load_model

# logging.basicConfig(level=logging.DEBUG)
        
        
def rollout(env, policy, max_steps=np.inf, video=False):
    if video:
        video_frames = []
    
    obs, done = env.reset(), False
    total_reward, steps = 0.0, 0.0
    
    while not done and steps <= max_steps:
        if video:
            video_frames.append(obs["pov"])
        
        action = policy.predict(torch.tensor(obs["pov"]).float())
        obs, reward, done, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
    if video:
        imageio.mimwrite(f"videos/rollout{str(uuid.uuid4())}.mp4", video_frames, fps=30.0)
        
    return total_reward


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
    # env = FrameSkipWrapper(gym.make("MineRLTreechop-v0"))
    env = FrameStackWrapper(gym.make("MineRLTreechop-v0"))
    env.make_interactive(port=6666, realtime=True)
    
    model = load_model("models/model_stack4_BCE_10v0.0")
    
    with torch.no_grad():
        run_reward = rollout(env, model, video=True)
    
    # models = [load_model(path) for path in ["models/model_rgb_BCE_5v0.0", "models/model_rgb_BCE_50v0.0"]]
    # validate_policy(env, models, max_steps=200)
    

if __name__ == "__main__":
    main()