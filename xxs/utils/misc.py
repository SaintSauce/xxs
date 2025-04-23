import os
import random
import torch
import numpy as np
from huggingface_hub import login
from dotenv import load_dotenv

def set_seed(seed: int):
    """ set random seed (reproducibility) """

    print(f"Setting seed to {seed}.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def hf_login():
    """ login to huggingface """

    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment")

    login(
        token=hf_token, 
        add_to_git_credential=True
    )