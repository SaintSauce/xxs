import os
import math
import logging
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

from xxs.utils.data import (
    load_split_dataset_from_hf,
    format_cot_prompt,
    get_dataloader,
    extract_gold_answer,
    extract_predicted_answer
)
from xxs.evaluation.eval_model import ModelEvaluator

class PPOTrainer:
    """ ppo trainer """

    def __init__(self, config, device: torch.device):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # store config and device
        self.config = config
        self.device = device

        self.logger.info(f"Initializing PPO trainer with device: {device}")

        # RL data parameters
        self.dataset_name = config.get("dataset_name")
        self.seed = int(config.get("seed"))
        self.ratios = config.get("split_ratios")

        # generation settings
        self.max_length = int(config.get("max_length"))
        self.gen_max_new_tokens = int(config.get("gen_max_new_tokens", 128))

        # PPO hyperparameters
        self.batch_size = int(config.get("rl_batch_size"))
        self.ppo_epochs = int(config.get("ppo_epochs", 4))
        self.clip_eps = float(config.get("ppo_clip_eps", 0.2))
        self.gamma = float(config.get("gamma", 0.99))
        self.lam = float(config.get("gae_lambda", 0.95))
        self.beta = float(config.get("kl_coef", 0.1))
        self.alpha = float(config.get("verifier_coef", 0.5))
        self.max_updates = int(config.get("max_updates", 1000))
        self.save_interval = int(config.get("save_interval", 50))

        # output directory
        raw_dir = config.get("ppo_output_dir", "ppo_ckpt")
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        self.output_dir = os.path.join(repo_root, raw_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # load policy and reference from local SFT checkpoint
        sft_dir = config.get("sft_output_dir")
        print(f"Loading SFT checkpoint from {sft_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            sft_dir, 
            local_files_only=True, 
            trust_remote_code=True
        )

        if self.tokenizer.pad_token_id is None: 
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        self.policy = AutoModelForCausalLM.from_pretrained(
            sft_dir, 
            local_files_only=True, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        self.reference = AutoModelForCausalLM.from_pretrained(
            sft_dir, 
            local_files_only=True, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(self.device)

        self.reference.eval()

        # optimizer and scheduler
        lr = float(config.get("rl_learning_rate", 1e-5))
        
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        
        total_steps = self.max_updates

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(config.get("rl_warmup_steps", 200)),
            num_training_steps=total_steps
        )

        # placeholder for data loader and gold answers
        self.loader = None
        self.gold_answers = []

        # metrics history
        self.metrics = defaultdict(list)

    def prepare_data(self):
        """ load RL prompts and build DataLoader of raw CoT prompts """
        self.logger.info(f"Loading dataset: {self.dataset_name}")
        
        splits = load_split_dataset_from_hf(
            dataset_name=self.dataset_name,
            seed=self.seed,
            sft_ratio=self.ratios["sft"],
            rl_ratio=self.ratios["rl"],
            val_ratio=self.ratios["val"]
        )
        rl_raw = splits["rl_train"]
        self.logger.info(f"Loaded {len(rl_raw)} examples for RL training")

        self.gold_answers = [
            extract_gold_answer(ex["answer"].strip()) for ex in rl_raw
        ]

        def prep_rl(ex):
            prompt = format_cot_prompt(ex["question"])
            
            # pad / truncate to a fixed length so every sample in a batch is equal-sized
            return self.tokenizer(
                prompt,
                truncation=True,            # cut off if it is longer than max_length
                padding="max_length",       # pad with PAD-/EOS-token up to max_length
                max_length=self.max_length, # 512 in your config
                return_tensors="pt"
            )

        rl_ds = rl_raw.map(prep_rl, remove_columns=rl_raw.column_names)
        rl_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

        self.loader = get_dataloader(
            rl_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=int(self.config.get("num_workers"))
        )

    def _generate_batch(self, batch):
        """ generate CoT sequences and compute log-probs of those sequences """

        b = {
            k: v.squeeze(1).to(self.device) for k, v in batch.items()
        }

        self.policy.eval()

        with torch.no_grad():
            seqs = self.policy.generate(
                **b,
                do_sample=False,
                max_new_tokens=self.gen_max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=False,
            )
        
        if (seqs != self.tokenizer.pad_token_id).sum() == 0:
            raise ValueError("all-PAD sequence – regenerate")
        
        # stale log-probabilities (no grad) for PPO ratio
        old_logp = self._log_prob(self.policy, seqs).detach()

        self.policy.train()

        return seqs, old_logp
    
    def _log_prob(self, model, seqs):
        """ compute total log-probability of `seqs` under `model` """

        labels = seqs.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # forward pass in fp32 for extra stability
        with torch.autocast(device_type=self.device.type, enabled=False):
            out = model(seqs, labels=labels)

        valid_len = (labels != -100).sum(dim=1)
        logp = -out.loss * valid_len         # (B,)

        logp = torch.nan_to_num(logp, neginf=-1e4, posinf=1e4)

        return logp

    def _compute_rewards(self, seqs, old_logp):
        """ compute final-answer reward + verifier bonus - KL penalty """
        
        final_rewards = []
        
        # compute final-answer reward
        for s, gold in zip(seqs, self.gold_answers):
            txt = self.tokenizer.decode(s, skip_special_tokens=True)
            pred = extract_predicted_answer(txt)
            final_rewards.append(1.0 if pred == gold else 0.0)
        
        final_rewards = torch.tensor(final_rewards, device=self.device)
        self.logger.debug(f"Average final answer reward: {final_rewards.mean().item():.3f}")

        verifier_bonus = []
        
        # verifier loaded in train()
        for idx, s in enumerate(seqs):
            txt = self.tokenizer.decode(s, skip_special_tokens=True)
            cot = txt.split("A:")[-1]
            steps = [step.strip() for step in cot.split('.') if step.strip()]
            enc = self.verifier_tok(
                steps,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.verifier(**enc).logits
                probs = torch.softmax(logits, dim=-1)[:, 1]

            verifier_bonus.append(probs.mean())
        
        verifier_bonus = torch.stack(verifier_bonus).to(self.device) * self.alpha

        with torch.no_grad():
            kl_terms = (old_logp - self._ref_logp(seqs)).detach()
            kl = kl_terms.mean() * self.beta

            if torch.isnan(kl):                     # just in case
                kl = torch.zeros_like(kl)

        self.metrics['answer_acc'].append(final_rewards.mean().item())
        self.metrics['verifier_bonus'].append(verifier_bonus.mean().item())
        self.metrics['kl_penalty'].append(kl.item())

        return final_rewards + verifier_bonus - kl

    def _ref_logp(self, seqs):
        with torch.no_grad():
            return self._log_prob(self.reference, seqs).detach()

    def _plot(self):
        
        # plot reward components over updates
        updates = range(len(self.metrics['total_reward']))
        
        plt.figure(figsize=(8, 5))
        plt.plot(updates, self.metrics['total_reward'], label='Total Reward')
        plt.plot(updates, self.metrics['answer_acc'], label='Answer Acc')
        plt.plot(updates, self.metrics['verifier_bonus'], label='Verifier Bonus')
        plt.plot(updates, self.metrics['kl_penalty'], label='KL Penalty')
        plt.xlabel('Update')
        plt.ylabel('Value')
        plt.title('Reward Components Over Updates')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'reward_components.png'))
        plt.close()

    def train(self):
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        
        self.logger.info("Starting PPO training")
        self.logger.info(f"Training for {self.max_updates} updates")
        self.logger.info(f"Batch size: {self.batch_size}, PPO epochs: {self.ppo_epochs}")

        verifier_dir = self.config.get("verifier_dir")
        self.logger.info(f"Loading verifier from {verifier_dir}")

        self.verifier_tok = DistilBertTokenizerFast.from_pretrained(
            verifier_dir,   
            local_files_only=True
        )

        self.verifier = DistilBertForSequenceClassification.from_pretrained(
            verifier_dir, 
            local_files_only=True
        ).to(self.device)
        
        self.verifier.eval()

        self.prepare_data()

        data_iter = iter(self.loader)

        for update in range(self.max_updates):

            # refresh iterator only when exhausted
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.loader)
                batch = next(data_iter)

            # generate CoT sequences and compute log-probs of those sequences
            seqs, old_logp = self._generate_batch(batch)

            if torch.isnan(old_logp).any() or torch.isinf(old_logp).any():
                print(f"[warn] skipped bad batch at update {update}")
                continue        # jumps to the next update

            old_logp = old_logp.detach()

            # compute final-answer reward + verifier bonus - KL penalty
            rewards = self._compute_rewards(seqs, old_logp)

            # compute returns and advantages
            returns = rewards.flip(0).cumsum(0).flip(0)
            advantages = returns - returns.mean()

            # update metrics
            self.metrics['total_reward'].append(rewards.mean().item())
            
            if update % 10 == 0:
                self.logger.info(
                    f"Update {update}/{self.max_updates} - "
                    f"Total Reward: {rewards.mean().item():.3f} - "
                    f"Answer Acc: {self.metrics['answer_acc'][-1]:.3f} - "
                    f"Verifier Bonus: {self.metrics['verifier_bonus'][-1]:.3f} - "
                    f"KL Penalty: {self.metrics['kl_penalty'][-1]:.3f}"
                )

            self.policy.train()

            # PPO update loop
            for _ in range(self.ppo_epochs):
                # differentiable log-prob under the *current* policy
                new_logp = self._log_prob(self.policy, seqs)   # <<< NEW

                log_ratio = (new_logp - old_logp).clamp(-20, 20)  # keeps exp() finite
                ratio = log_ratio.exp()

                clipped = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
                obj = torch.min(ratio * advantages, clipped * advantages)
                loss = -obj.mean()

                if torch.isnan(loss) or torch.isinf(loss):      # 🆕 skip poisoned batch
                    self.logger.warning("NaN/Inf loss – batch skipped")
                    continue

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                self.metrics['surrogate_loss'].append(loss.item())

            # save model
            if update % self.save_interval == 0:
                ckpt = os.path.join(self.output_dir, f"update_{update}")
                self.logger.info(f"Saving checkpoint at update {update} to {ckpt}")
                self.policy.save_pretrained(
                    ckpt,
                    safe_serialization=False
                )
                self.tokenizer.save_pretrained(
                    ckpt,
                    safe_serialization=False
                )

        # save final model
        self.logger.info("Saving final model")
        self.policy.save_pretrained(
            self.output_dir,
            safe_serialization=False
        )
        self.tokenizer.save_pretrained(
            self.output_dir,
            safe_serialization=False
        )

        # evaluate final model
        self.logger.info("Evaluating final model")
        evaluator = ModelEvaluator(
            config=self.config,
            device=self.device,
            model=self.policy,
            tokenizer=self.tokenizer
        )
        test_res = evaluator.evaluate(num_samples=0)
        self.logger.info(f"Final Test Accuracy: {test_res['test_accuracy']:.2f}%")

        self._plot()
        self.logger.info("Training completed successfully")