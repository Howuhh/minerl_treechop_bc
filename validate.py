import gym
import tqdm
import uuid
import torch
import minerl
import logging

import numpy as np

from tabulate import tabulate

from model import ConvNetRGB
from wrappers import FrameSkipWrapper, FrameStackWrapper, GreyScaleWrapper
from utils import load_model

logging.basicConfig(level=logging.DEBUG)
        

def rollout(env, policy, max_steps=np.inf, video=False, seed=None):
    if video:
        env = gym.wrappers.monitor.Monitor(env, f"videos/{policy.name}/rollout_{str(uuid.uuid4())}", resume=True)

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
    
    # TODO: add tqdm
    for policy, env in policies_with_envs:
        policy_rewards = [rollout(env, policy, **kwargs) for _ in range(n_evals)]
        
        table.append([policy.name, np.mean(policy_rewards), np.std(policy_rewards)])

    print(tabulate(table, headers="firstrow", tablefmt="github"))


def main_validate():
    env = gym.make("MineRLTreechop-v0")
    
    # policy_env = (
    #     (load_model("models/model_stack1_BCE_50_1200"), env),
    #     (load_model("models/model_stack2_BCE_50_1200"), FrameStackWrapper(env, 2)),
    #     (load_model("models/model_stack4_BCE_50_1200"), FrameStackWrapper(env, 4)),
    #     (load_model("models/stack1_BCE_50_1200_grey"), GreyScaleWrapper(env)),
    #     (load_model("models/stack2_BCE_50_1200_grey"), FrameStackWrapper(GreyScaleWrapper(env), 2)),   
    #     (load_model("models/stack4_BCE_50_1200_grey"), FrameStackWrapper(GreyScaleWrapper(env), 4)), 
    # )
    policy_env = (
        (load_model("models/stack2_BCE_20_1200_rgb"), FrameSkipWrapper(FrameStackWrapper(env, 2))),
        (load_model("models/stack2_BCE_20_1200_rgb"), FrameStackWrapper(FrameSkipWrapper(env), 2))
    )
    
    with torch.no_grad():
        validate_policy(policy_env, n_evals=5, max_steps=500, seed=42)


def main():
    env = FrameSkipWrapper(FrameStackWrapper(gym.make("MineRLTreechop-v0"), 2))
    env.make_interactive(port=6666, realtime=True)
    
    model = load_model("models/stack2_BCE_20_1200_rgb")
    
    with torch.no_grad():
        run_reward = rollout(env, model, video=True)
    

if __name__ == "__main__":
    main()
    # main_validate()