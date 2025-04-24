import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from xxs.utils.data import (
    load_split_dataset_from_hf,
    format_cot_prompt,
    combine_prompt_answer,
    get_dataloader
)
from xxs.utils.config import ConfigLoader
from xxs.models.load_model import HFModelLoader

class PPOTrainer:
    """ PPO trainer for RL fine-tuning """
    
    def __init__(self, config: ConfigLoader, device: torch.device):
        
        # load config fields
        self.dataset_name   = config.get("dataset_name")
        self.model_name     = config.get("model_name")
        self.ratios         = config.get("split_ratios")
        self.seed           = int(config.get("seed"))
        self.max_length     = int(config.get("max_length"))
        self.batch_size     = int(config.get("batch_size"))
        self.num_workers    = int(config.get("num_workers"))
        self.lr             = float(config.get("learning_rate"))
        self.weight_decay   = float(config.get("weight_decay"))
        self.num_epochs     = int(config.get("num_epochs"))
        self.grad_accum     = int(config.get("grad_accum_steps"))
        self.warmup_steps   = int(config.get("warmup_steps"))
        
        # PPO specific parameters
        self.ppo_epochs     = int(config.get("ppo_epochs", 4))
        self.clip_epsilon   = float(config.get("clip_epsilon", 0.2))
        self.gamma          = float(config.get("gamma", 0.99))
        self.lam            = float(config.get("lam", 0.95))
        self.entropy_coef   = float(config.get("entropy_coef", 0.01))
        self.value_coef     = float(config.get("value_coef", 0.5))

        raw_output_dir      = config.get("ppo_output_dir", "ppo_checkpoint")
        repo_root           = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..","..","..")
        )
        self.output_dir     = os.path.join(repo_root, raw_output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.device         = device
        self.model_loader   = HFModelLoader(self.model_name, device)

        # placeholders
        self.tokenizer      = None
        self.loader         = None
        self.val_loader     = None
        self.val_answers    = None
        self.model          = None
        self.old_model      = None  # for PPO

        # metrics history
        self.epochs         = []
        self.train_steps    = []
        self.train_loss     = []
        self.val_loss       = []
        self.val_acc        = []
        self.returns        = []  # for tracking returns
        self.advantages     = []  # for tracking advantages

    def _verify_save(self, save_dir: str) -> bool:
        required = [
            "config.json", "pytorch_model.bin",
            "tokenizer_config.json", "vocab.json",
            "merges.txt", "special_tokens_map.json"
        ]
        
        for fn in required:
            if not os.path.exists(os.path.join(save_dir, fn)):
                print(f"Warning: missing {fn} in {save_dir}")
                return False
        
        return True

    def prepare_data(self):
        """ load model & tokenizer, split data, and build loaders """
        
        self.model, self.tokenizer = self.model_loader.load()
        self.old_model = self.model_loader.load()[0]  # create copy for PPO

        # splits
        splits = load_split_dataset_from_hf(
            dataset_name=self.dataset_name,
            seed=self.seed,
            sft_ratio=self.ratios["sft"],
            rl_ratio=self.ratios["rl"],
            val_ratio=self.ratios["val"]
        )
        rl_raw = splits["rl_train"]
        val_raw = splits["val"]

        # RL loader
        def prep_rl(ex):
            prompt = format_cot_prompt(ex["question"])
            return self.tokenizer(
                prompt, truncation=True, padding="max_length",
                max_length=self.max_length, return_tensors="pt"
            )

        rl_ds = rl_raw.map(prep_rl, remove_columns=rl_raw.column_names)
        rl_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        self.loader = get_dataloader(
            rl_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=True, drop_last=True
        )

        # val loader (for accuracy)
        def prep_val(ex):
            prompt = format_cot_prompt(ex["question"])
            return self.tokenizer(
                prompt, truncation=True, padding="max_length",
                max_length=self.max_length, return_tensors="pt"
            )

        val_ds = val_raw.map(prep_val, remove_columns=val_raw.column_names)
        val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        self.val_loader = get_dataloader(
            val_ds, batch_size=1,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True
        )

        # gold answers for accuracy
        self.val_answers = [ex["answer"].strip() for ex in val_raw]

    def compute_returns_and_advantages(self, rewards, values, dones):
        """ compute returns and advantages using GAE """
        
        returns = []
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_done = 1
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            gae = delta + self.gamma * self.lam * (1 - next_done) * gae
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)
        
        return torch.tensor(returns), torch.tensor(advantages)

    def ppo_loss(self, logprobs, old_logprobs, advantages, values, returns):
        """ compute PPO loss """
        
        ratio = torch.exp(logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = 0.5 * (returns - values).pow(2).mean()
        
        entropy_loss = -logprobs.mean()
        
        return policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

    @torch.no_grad()
    def evaluate(self):
        """ return (val_loss, val_accuracy) for current model """
        
        self.model.eval()
        
        # exact-match accuracy
        correct = 0
        
        for i, batch in enumerate(self.val_loader):
            b = {k: v.squeeze(1).to(self.device) for k,v in batch.items()}
            out_ids = self.model.generate(**b, max_new_tokens=128)
            txt = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            pred_ans = txt.strip().split()[-1]
            gold_ans = self.val_answers[i].split()[-1]
            
            if pred_ans == gold_ans:
                correct += 1
        
        acc = correct / len(self.val_loader) * 100
        self.model.train()
        return 0.0, acc  # No loss computation in PPO evaluation

    def _save_plots(self):
        """ plot & save training metrics """
        
        # loss curve
        plt.figure(figsize=(8,6))
        plt.plot(self.train_steps, self.train_loss, label="Train Loss")
        plt.xlabel("Update Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("PPO Loss Curve")
        plt.savefig(os.path.join(self.output_dir, "loss_curve.png"))
        plt.close()

        # returns curve
        plt.figure(figsize=(8,6))
        plt.plot(self.train_steps, self.returns, label="Returns")
        plt.xlabel("Update Step")
        plt.ylabel("Returns")
        plt.title("PPO Returns")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "returns_curve.png"))
        plt.close()

        # accuracy curve
        plt.figure(figsize=(8,6))
        plt.plot(self.epochs, self.val_acc, marker="o", label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "accuracy_curve.png"))
        plt.close()

    def train(self):
        """ run PPO training with per-epoch validation and final plotting """
        
        optimizer = AdamW(self.model.parameters(),
                         lr=self.lr,
                         weight_decay=self.weight_decay)
        
        total_steps = math.ceil(len(self.loader)/self.grad_accum) * self.num_epochs
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        global_step = 0

        for epoch in range(1, self.num_epochs + 1):
            self.epochs.append(epoch)
            running_loss = 0.0

            for step, batch in enumerate(self.loader, start=1):
                b = {k: v.to(self.device) for k,v in batch.items()}
                
                # generate responses and compute rewards
                with torch.no_grad():
                    self.old_model.eval()
                    old_outputs = self.old_model(**b)
                    old_logprobs = old_outputs.logits.log_softmax(dim=-1)
                    old_values = old_outputs.value
                
                self.model.train()
                outputs = self.model(**b)
                logprobs = outputs.logits.log_softmax(dim=-1)
                values = outputs.value
                
                # compute rewards (placeholder - implement your reward function)
                rewards = torch.ones_like(values)  # replace with actual reward computation
                dones = torch.zeros_like(rewards)
                
                # compute returns and advantages
                returns, advantages = self.compute_returns_and_advantages(
                    rewards, values, dones
                )
                
                # normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # PPO loss
                loss = self.ppo_loss(logprobs, old_logprobs, advantages, values, returns)
                loss = loss / self.grad_accum
                loss.backward()
                
                running_loss += loss.item()
                
                if step % self.grad_accum == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    self.train_steps.append(global_step)
                    self.train_loss.append(running_loss / self.grad_accum)
                    self.returns.append(returns.mean().item())
                    self.advantages.append(advantages.mean().item())
                    
                    running_loss = 0.0
                    
                    # update old model
                    self.old_model.load_state_dict(self.model.state_dict())
            
            # evaluate at end of epoch
            val_loss, val_acc = self.evaluate()
            self.val_loss.append(val_loss)
            self.val_acc.append(val_acc)
            
            print(f"Epoch {epoch}: Val Acc = {val_acc:.2f}%")
        
        # save final model and plots
        self.model.save_pretrained(self.output_dir)
        self._save_plots()
