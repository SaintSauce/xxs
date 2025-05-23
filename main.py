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
    parser = argparse.ArgumentParser(description="Pipeline entrypoint - EDA, SFT, PPO, GRPO")
    parser.add_argument(
        "--mode",
        choices=["eda", "sft", "ppo", "grpo"],
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
        print(f"Exploring dataset: {dataset_name}")
        print(f"Using model: {model_name}")

        print("Loading dataset...")

        df = setup_df(model_name)

        print("Dataset loaded successfully")
        print(f"Dataset stats: {df.describe()}")

        print("Plotting character length distribution...")

        # chars stats
        plot_char_length_distribution(df, question=True)
        plot_char_length_distribution(df, question=False)

        print("Plotting token length distribution...")

        # tokens stats
        plot_token_length_distribution(df, question=True)
        plot_token_length_distribution(df, question=False)

        print("EDA completed!")

    elif args.mode == "sft":
        from xxs.train.train import run_sft
        run_sft(config, device)
    elif args.mode == "ppo":
        from xxs.train.train import run_ppo
        run_ppo(config, device)
    elif args.mode == "grpo":
        from xxs.train.train import run_grpo
        run_grpo(config, device)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
if __name__ == "__main__":
    main()