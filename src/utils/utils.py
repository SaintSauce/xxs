import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Union, Dict, List
from transformers import PreTrainedTokenizer

"""

BASICS

"""

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

"""

EXPLORATION

"""

def get_text_lengths(texts: List[str]) -> np.ndarray:

    """
    
    compute char length for a list of texts
    
    """
    
    return np.array([len(t) for t in texts])


def get_token_lengths(
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
) -> np.ndarray:
    """
    
    compute token length for a list of texts
    
    """

    lengths = []
    for t in texts:
        toks = tokenizer(t, add_special_tokens=False)
        lengths.append(len(toks["input_ids"]))
    return np.array(lengths)

"""

PLOTTING

"""

def plot_histogram(
    data: Sequence[Union[int, float]],
    bins: int = 50,
    title: str = None,
    xlabel: str = None,
    ylabel: str = "Count",
    figsize: tuple = (8, 4),
    show: bool = True
):
    """

    plot a basic histogram of numeric data

    """

    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, edgecolor='black')

    if title:
        plt.title(title)

    if xlabel:
        plt.xlabel(xlabel)
    
    if ylabel:
        plt.ylabel(ylabel)
    
    if show:
        plt.show()

def plot_bar(
    stats: Dict[str, float],
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: tuple = (8, 4),
    show: bool = True
):
    """

    plot a bar chart from a dictionary of label -> value

    """

    labels = list(stats.keys())
    values = list(stats.values())
    x = np.arange(len(labels))

    plt.figure(figsize=figsize)
    plt.bar(x, values, edgecolor='black')
    plt.xticks(x, labels, rotation=45, ha='right')

    if title:
        plt.title(title)

    if xlabel:
        plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)

    plt.tight_layout()
    
    if show:
        plt.show()
