import random
import torch
import numpy as np

def set_seed(seed: int):
    
    """

    set random seed (reproducibility)
    
    """

    print(f"Setting seed to {seed}.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""

FORMATTING

"""

def format_cot_prompt(
    question: str, 
    cot_prefix: str = "Let's think step by step."
):
    
    """

    format the question w CoT prefix
    
    """

    return f"Q: {question.strip()}\nA: {cot_prefix}"

def combine_prompt_answer(prompt: str, answer: str) -> str:

    """

    concatenate a CoT prompt w answer
    
    """

    return f"{prompt} {answer.strip()}"