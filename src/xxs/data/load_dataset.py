from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader
from typing import Tuple

from src.xxs.utils.data import (
    load_split_dataset_from_hf,
    format_cot_prompt, 
    combine_prompt_answer, 
    get_dataloader
)

class HFDatasetLoader:
    """ big mac class for the dataset """

    def __init__(
        self, 
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        ratios: dict[str, float],
        max_length: int,
        seed: int,
        sft_batch_size: int,
        rl_batch_size: int,
        num_workers: int,
    ):
        self.dataset_name       =       dataset_name
        self.tokenizer          =       tokenizer
        self.ratios             =       ratios
        self.max_length         =       max_length
        self.seed               =       seed
        self.sft_batch_size     =       sft_batch_size
        self.rl_batch_size      =       rl_batch_size
        self.num_workers        =       num_workers
    
    def load(self) -> Tuple[DataLoader, DataLoader, DataLoader]:

        # using util fn
        splits = load_split_dataset_from_hf(
            dataset_name="gsm8k",
            splits=self.ratios,
            seed=self.seed
        )
        
        sft_raw = splits["sft"]
        rl_raw  = splits["rl"]
        val_raw = splits["val"]

        # preprocessing for SFT
        def prep_sft(sample):
            prompt = format_cot_prompt(sample["question"])
            answer = sample["answer"].strip()

            p_tok = self.tokenizer(prompt, add_special_tokens=False)
            p_len = len(p_tok["input_ids"])
            
            full = combine_prompt_answer(prompt, answer, self.tokenizer.eos_token)
            
            tk = self.tokenizer(
                full,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            
            labels = tk["input_ids"].copy()
            labels[:p_len] = -100
            tk["labels"] = labels
            
            return tk

        # preprocessing for RL: prompt only
        def prep_rl(sample):

            # format the question w CoT prefix
            prompt = format_cot_prompt(sample["question"])

            # tokenize the prompt
            # padding to max length
            # truncation to max length
            tk = self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            
            return tk
    
        # apply preprocessing
        sft_ds = sft_raw.map(prep_sft, remove_columns=sft_raw.column_names)
        rl_ds  = rl_raw.map(prep_rl,  remove_columns=rl_raw.column_names)
        val_ds = val_raw.map(prep_rl, remove_columns=val_raw.column_names)

        # convert to pytorch format
        sft_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        rl_ds.set_format(type="torch",  columns=["input_ids", "attention_mask"])
        val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # create dataloaders
        sft_loader = get_dataloader(
            sft_ds,
            batch_size=self.sft_batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        rl_loader = get_dataloader(
            rl_ds,
            batch_size=self.rl_batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        val_loader = get_dataloader(
            val_ds,
            batch_size=self.rl_batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        return sft_loader, rl_loader, val_loader