# MineRL TreeChop with Behavioural Cloning

## Run rollout with trained model

```python
import gym

from utils import load_model
from wrapper import FrameSkipWrapper

env = FrameSkipWrapper(gym.make("MineRLTreechop-v0"))
env.make_interactive(port=6666, realtime=True)
    
model = load_model("models/model_rgb_BCE_50v0.0", "cpu")

rollout(env, model)
```

Also, to interact with the agent during rollout:

```bash
python -m minerl.interactor 6666
```