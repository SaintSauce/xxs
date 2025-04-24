from datasets import load_dataset
from datasets import DatasetDict
from torch.utils.data import DataLoader
 
def load_split_dataset_from_hf(
    dataset_name: str,
    seed: int = 42,
    sft_ratio: float = 0.80,
    rl_ratio: float = 0.10,
    val_ratio: float = 0.10
)-> DatasetDict:
    
    # load the dataset
    ds = load_dataset(
        dataset_name, 
        "main"
    )

    # SFT training portion
    train_val = ds["train"].train_test_split(
        test_size=(1.0 - sft_ratio), 
        seed=seed
    )
    sft_train = train_val["train"]

    # RL training portion
    rem_split = train_val["test"].train_test_split(
        test_size=(val_ratio / (rl_ratio + val_ratio)),
        seed=seed
    )
    rl_train   = rem_split["train"]
    val = rem_split["test"]

    # official test split untouched
    test_set = ds["test"]

    return DatasetDict(
        {
            "sft_train": sft_train,
            "rl_train": rl_train,
            "val": val,
            "test": test_set
        }
    )

def format_cot_prompt(
    question: str, 
    cot_prefix: str = "Let's think step by step."
):
    """ wrap into CoT style prompt """

    return f"Q: {question.strip()}\nA: {cot_prefix}"

def combine_prompt_answer(prompt: str, answer: str, eos_token: str) -> str:
    """ concatenate a CoT prompt w answer """

    return f"{prompt} {answer.strip()}{eos_token}"

def get_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """ return pytorch dataloader """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )