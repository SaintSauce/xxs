import pandas as pd
from datasets import load_dataset
from transformers import PreTrainedTokenizer

def sample_inspection(
    df: pd.DataFrame, 
    n_samples: int, 
    seed: int
) -> pd.DataFrame:
    """ return n random samples from the dataset """

    return df.sample(n=n_samples, random_state=seed)[["question", "answer"]]

def clean_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """ strip leading/trailing whitespace from text fields """
    
    df = df.copy()
    
    df['question'] = df['question'].astype(str).str.strip()
    df['answer'] = df['answer'].astype(str).str.strip()
    
    return df

def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """ remove missing values from the dataset """
    
    # replace empty strings with NaN
    df = df.copy()
    
    df.replace(
        {'': pd.NA}, 
        inplace=True
    )

    # drop rows with NaN values in the question or answer columns
    return df.dropna(subset=['question', 'answer'])


def drop_duplicate(df: pd.DataFrame) -> pd.DataFrame:
    """ remove duplicate samples from the dataset """

    df = df.copy()
    
    # remove duplicates based on question and answer
    return df.drop_duplicates(subset=['question', 'answer'])

def load_dataset_split(
    dataset_name: str = "openai/gsm8k",
    split: str = "train", 
    clean: bool = True
) -> pd.DataFrame:
    """ load dataset and return as pandas df for the given split """

    ds = load_dataset(
        dataset_name, 
        "main"
    )
    df = ds[split].to_pandas()
    
    if clean:
        df = clean_whitespace(df)
        df = drop_missing(df)
        df = drop_duplicate(df)

    return df

def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """ add word count features to the dataset """

    # count the number of words in the question and answer
    df = df.copy()
    df['q_len_words'] = df['question'].str.split().str.len()
    df['a_len_words'] = df['answer'].str.split().str.len()
    
    return df

def add_char_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """ add character count features to the dataset """
    
    # count the number of characters in the question and answer
    df = df.copy()
    df['q_len_chars'] = df['question'].astype(str).str.len()
    df['a_len_chars'] = df['answer'].astype(str).str.len()
    
    return df

def add_token_length_features(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer
) -> pd.DataFrame:
    """ add token-count features using a HF tokenizer """
    
    df = df.copy()
    
    # question
    df['q_len_tokens'] = df['question'].astype(str).apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=False))
    )
    
    # answer
    df['a_len_tokens'] = df['answer'].astype(str).apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=False))
    )
    
    return df