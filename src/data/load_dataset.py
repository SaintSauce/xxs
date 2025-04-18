from datasets import load_dataset, DatasetDict
from typing import Tuple
import torch
from transformers import PreTrainedTokenizer
from utils.utils import format_cot_prompt, combine_prompt_answer

class HFDatasetLoader:
    """

    Load a dataset from HF

    """

    def __init__(
        self, 
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        split_ratio: float = 0.2,           # rl train dataset split ratio
        max_length: int = 512,
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
    
    def load(self):

        # Dataset
        dataset: DatasetDict = load_dataset(
            self.dataset_name,
            "main"
        )

        return dataset