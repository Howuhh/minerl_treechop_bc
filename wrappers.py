import gym
import minerl
import imageio
import numpy as np

from collections import deque


def rgb_to_grey(img):
    """
    Convert image from rgb to grey scale.

    Parameters
    ----------
    img: np.ndarray, shape [batch, H, W, 3]
        Single image or batch of images.

    Returns
    -------
    np.ndarray, shape [batch, H, W, 1]
        Grey scaled image or batch of images. 
    
    """
    rgb_to_grey_vec = np.array([0.2989, 0.5870, 0.1140]).reshape(-1, 1)
    
    return img @ rgb_to_grey_vec


class GreyScaleWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return {'pov': rgb_to_grey(obs['pov'])}, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return {'pov': rgb_to_grey(obs['pov'])}


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
    def __init__(self, env, stack=4, ch_num=3):
        super().__init__(env)        
        self.stack = stack
        self.frame_stack = deque([], maxlen=stack)
        
        self.observation_space = gym.spaces.Dict({
            "pov": gym.spaces.Box(low=0, high=255, shape=(64, 64, ch_num * stack))
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
    def __init__(self, data_dir, stack=1, greyscale=False):
        self.data = minerl.data.make("MineRLTreechop-v0", data_dir=data_dir)
        
        self.stack = stack
        self.frame_stack = deque([], maxlen=stack)
        self.greyscale = greyscale
        
    def _stack_to_obs(self):
        return np.concatenate(self.frame_stack, axis=2)   
        
    def seq_iter(self, seq_len):
        for state, action, _, _, _ in self.data.batch_iter(batch_size=1, num_epochs=1, seq_len=seq_len):
            state = rgb_to_grey(state["pov"][0]) if self.greyscale else state["pov"][0]
            
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