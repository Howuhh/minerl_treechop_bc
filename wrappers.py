import gym
import minerl
import numpy as np

from collections import deque


def to_grey_scale(img):
    if isinstance(img, dict):
        r, g, b = img['pov'][:, :, 0], img['pov'][:, :, 1], img['pov'][:, :, 2]
        grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return {'pov': grey}
    else:
        r, g, b = img[:, :, :, 0], img[:, :, :, 1], img[:, :, :, 2]
        grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return grey


class GreyScaleWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return to_grey_scale(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = to_grey_scale(obs)
        return obs


class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        
    def step(self, action):
        total_reward = 0.0
        
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            
            if done:
                break
        
        return obs, total_reward, done, info


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, stack=4):
        super().__init__(env)        
        self.stack = stack
        self.frame_stack = deque([], maxlen=stack)
        
        self.observation_space = gym.spaces.Dict({
            "pov": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3 * stack))
            })
        
    def _stack_to_obs(self):
        assert len(self.frame_stack) == self.stack, "stack smaller than desired"
        return {"pov": np.concatenate(self.frame_stack, axis=2)}
        
    def reset(self):
        obs = self.env.reset()
        
        for _ in range(self.stack):            
            self.frame_stack.append(obs["pov"])
            
        return self._stack_to_obs()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frame_stack.append(obs["pov"])
        
        return self._stack_to_obs(), reward, done, info
        

class TreeChopDataset:
    def __init__(self, data_dir, stack=1, grey_scale=False):
        self.data = minerl.data.make("MineRLTreechop-v0", data_dir=data_dir)
        
        self.stack = stack
        self.frame_stack = deque([], maxlen=stack)
        self.greyscale = grey_scale
        
    def _stack_to_obs(self):
        return np.concatenate(self.frame_stack, axis=2)   
        
    def seq_iter(self, seq_len):
        for state, action, _, _, _ in self.data.batch_iter(batch_size=1, num_epochs=1, seq_len=seq_len):
            state = to_grey_scale(state["pov"][0]) if self.greyscale else state["pov"][0]
            
            if self.stack > 1:
                new_state = np.zeros((*state.shape[:3], state.shape[3] * self.stack))  # [batch, 64, 64, 3] -> [batch, 64, 64, 3 * k]

                for _ in range(self.stack):
                    self.frame_stack.append(state[0])
                
                new_state[0, :, :, :] = self._stack_to_obs()
                for i in range(1, seq_len):
                    self.frame_stack.append(state[i])
                    new_state[i, :, :, :] = self._stack_to_obs()

                yield new_state, action
            else:
                yield state, action