import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from transformers import PreTrainedTokenizer
from utils.utils import get_text_lengths, get_token_lengths

def explore_qa_dataset(
    dataset_name: str,
    config_name: Optional[str] = None,
    splits: List[str] = ["train"],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    show_plots: bool = True,
) -> Dict[str, Dict[str, float]]:
    
    """

    do eda on a QA-style dataset
    
    """
    
    # load dataset
    if config_name:
        ds = load_dataset(dataset_name, config_name)
    else:
        ds = load_dataset(dataset_name)

    stats: Dict[str, Dict[str, float]] = {}

    for split in splits:
        data = ds[split]
        questions = [q.strip() for q in data["question"]]
        answers = data["answer"] if "answer" in data.column_names else None

        # char-based stats
        q_char = get_text_lengths(questions)

        split_stats = {
            "num_samples": len(questions),
            "q_char_min": float(q_char.min()),
            "q_char_max": float(q_char.max()),
            "q_char_mean": float(q_char.mean()),
            "q_char_median": float(np.median(q_char)),
        }
        
        if answers is not None:
            a_char = get_text_lengths([a.strip() for a in answers])
            split_stats.update({
                "a_char_min": float(a_char.min()),
                "a_char_max": float(a_char.max()),
                "a_char_mean": float(a_char.mean()),
                "a_char_median": float(np.median(a_char)),
            })

        # token-based stats
        if tokenizer:
            q_tok = get_token_lengths(tokenizer, questions)

            split_stats.update({
                "q_tok_min": float(q_tok.min()),
                "q_tok_max": float(q_tok.max()),
                "q_tok_mean": float(q_tok.mean()),
                "q_tok_median": float(np.median(q_tok)),
            })
            
            if answers is not None:
                a_tok = get_token_lengths(tokenizer, [a.strip() for a in answers])
                split_stats.update({
                    "a_tok_min": float(a_tok.min()),
                    "a_tok_max": float(a_tok.max()),
                    "a_tok_mean": float(a_tok.mean()),
                    "a_tok_median": float(np.median(a_tok)),
                })

        stats[split] = split_stats

        if show_plots:
            # question char-length histogram
            plt.figure()
            plt.hist(q_char, bins=50)
            plt.title(f"{dataset_name}/{split} question lengths (chars)")
            plt.xlabel("Chars")
            plt.ylabel("Count")
            plt.show()

            if answers is not None:
                plt.figure()
                plt.hist(a_char, bins=50)
                plt.title(f"{dataset_name}/{split} answer lengths (chars)")
                plt.xlabel("Chars")
                plt.ylabel("Count")
                plt.show()

            if tokenizer:
                plt.figure()
                plt.hist(q_tok, bins=50)
                plt.title(f"{dataset_name}/{split} question lengths (tokens)")
                plt.xlabel("Tokens")
                plt.ylabel("Count")
                plt.show()

                if answers is not None:
                    plt.figure()
                    plt.hist(a_tok, bins=50)
                    plt.title(f"{dataset_name}/{split} answer lengths (tokens)")
                    plt.xlabel("Tokens")
                    plt.ylabel("Count")
                    plt.show()

    return stats