import pandas as pd
from xxs.utils.plot import plot_histogram
from transformers import AutoTokenizer

from xxs.utils.explore import (
    load_dataset_split,
    add_char_length_features,
    add_token_length_features
)

def setup_df(model_name: str) -> pd.DataFrame:
    """ load dataset and add features """
    
    print(f"Loading dataset with model: {model_name}")
    
    df = load_dataset_split()
    
    print("Dataset loaded successfully")
    print("Adding character length features...")
    
    df = add_char_length_features(df)
    
    print("Adding token length features...")
    print(f"Loading tokenizer for model: {model_name}")
    
    df = add_token_length_features(df, AutoTokenizer.from_pretrained(model_name))
    
    print("Features added successfully")
    
    return df

def plot_question_length_distribution(
    df: pd.DataFrame,
    col: str = 'q_len_words',
    bins: int = 30
) -> None:
    """ histogram of question word counts """
    
    plot_histogram(
        data=df[col],
        bins=bins,
        title=f"Question Length Distribution ({col})",
        xlabel="Word Count",
    )

def plot_answer_length_distribution(
    df: pd.DataFrame,
    col: str = 'a_len_words',
    bins: int = 30
) -> None:
    """ histogram of answer word counts """

    plot_histogram(
        data=df[col],
        bins=bins,
        title=f"Answer Length Distribution ({col})",
        xlabel="Word Count"
    )


def plot_char_length_distribution(
    df: pd.DataFrame,
    question: bool = True,
    bins: int = 30
) -> None:
    """ plot character-length distribution for questions or answers """

    col = 'q_len_chars' if question else 'a_len_chars'
    
    plot_histogram(
        data=df[col],
        bins=bins,
        title=f"Character Length Distribution ({'Question' if question else 'Answer'})",
        xlabel="Character Count"
    )


def plot_token_length_distribution(
    df: pd.DataFrame,
    question: bool = True,
    bins: int = 30
) -> None:
    """ plot token-length distribution for questions or answers """

    col = 'q_len_tokens' if question else 'a_len_tokens'
    
    plot_histogram(
        data=df[col],
        bins=bins,
        title=f"Token Length Distribution ({'Question' if question else 'Answer'})",
        xlabel="Character Count"
    )