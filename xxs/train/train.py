import torch
from xxs.utils.config import ConfigLoader

from xxs.sft.sft_trainer import SFTTrainer
# from xxs.rl.ppo_trainer import PPOTrainer
# from xxs.rl.grpo_trainer import GRPOTrainer

def run_sft(config: ConfigLoader, device: torch.device):
    """ run supervised finetuning """

    print("Running SFT...")
    trainer = SFTTrainer(config, device)
    trainer.prepare_data()
    trainer.train()

# def run_ppo(config: ConfigLoader, device: torch.device):
#     """ run PPO training """

#     print("Running PPO...")
#     trainer = PPOTrainer(config, device)
#     trainer.prepare_data()
#     trainer.train()

# def run_grpo(config: ConfigLoader, device: torch.device):
#     """ run GRPO training """

#     print("Running GRPO...")
#     trainer = GRPOTrainer(config, device)
#     trainer.prepare_data()
#     trainer.train()