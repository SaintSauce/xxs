from xxs.utils.config import ConfigLoader
from xxs.exploration.eda import (
    setup_df,
    plot_char_length_distribution, 
    plot_token_length_distribution
)
from xxs.utils.misc import set_seed

import os
import torch
import argparse
from xxs.utils.misc import hf_login

def main():
    parser = argparse.ArgumentParser(description="Pipeline entrypoint - EDA, SFT, PPO, GRPO, EVAL")
    parser.add_argument(
        "--mode",
        choices=["eda", "sft", "ppo", "grpo", "eval"],
        required=True
    )

    args = parser.parse_args()

    # Load the config
    config = ConfigLoader("configs/config.yaml")
    seed = config.get("seed")
    dataset_name = config.get("dataset_name")
    model_name = config.get("model_name")

    print(f"HF_TOKEN: {os.getenv('HF_TOKEN')}")

    hf_login()

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "eda":
        df = setup_df(dataset_name)

        # chars stats
        plot_char_length_distribution(df, question=True)
        plot_char_length_distribution(df, question=False)

        # tokens stats
        plot_token_length_distribution(df, question=True)
        plot_token_length_distribution(df, question=False)

        # do more exploration

    # TODO: add sft
    elif args.mode == "sft":
        from xxs.train.train import run_sft
        run_sft(config, device)
    # TODO: add ppo
    elif args.mode == "ppo":
        from xxs.train.train import run_ppo
        run_ppo(config, device)
    # TODO: add grpo
    elif args.mode == "grpo":
        from xxs.train.train import run_grpo
        run_grpo(config, device)
    # TODO: add eval
    elif args.mode == "eval":
        from xxs.evaluation import run_evaluation
        run_evaluation(config, device)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")