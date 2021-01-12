# MineRL TreeChop with Behavioural Cloning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Howuhh/minerl_treechop_bc/blob/main/run_colab.ipynb)

## Run rollout with trained model

```python
import gym

from model import ConvNetRGB  # import needed for torch.load
from utils import load_model
from wrapper import FrameSkipWrapper

env = FrameSkipWrapper(gym.make("MineRLTreechop-v0"))
env.make_interactive(port=6666, realtime=True)
    
model = load_model("models/model_rgb_BCE_50v0.0", "cpu")

rollout(env, model)
```

Also, to interact with the agent during rollout (after the world is created):

```bash
python -m minerl.interactor 6666
```