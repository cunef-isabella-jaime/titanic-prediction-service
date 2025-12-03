import os
import random
import numpy as np

def set_seed(seed=None):
    if seed is None:
        env = os.getenv("SEED")
        seed = int(env) if env is not None else 42

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except:
        pass

    return seed
