import os, logging
from collections import defaultdict

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
)

from xxs.utils.data import (
    load_split_dataset_from_hf, format_cot_prompt, get_dataloader,
    extract_gold_answer, extract_predicted_answer
)
from xxs.evaluation.eval_model import ModelEvaluator

class GRPOTrainer:
    def __init__(self, config, device: torch.device):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialising GRPO trainer on {device}")

        self.cfg   = config
        self.device = device
        self.seed   = int(config.get("seed"))
        self.max_len = int(config.get("max_length"))
        self.gen_len = int(config.get("gen_max_new_tokens", 128))

        self.batch_size   = int(config.get("rl_batch_size"))
        self.lr           = float(config.get("rl_learning_rate", 1e-5))
        self.epochs       = int(config.get("grpo_epochs", 1))
        self.beta         = float(config.get("kl_coef", 0.1))
        self.alpha        = float(config.get("verifier_coef", 0.5))
        self.max_updates  = int(config.get("max_updates", 1000))
        self.save_every   = int(config.get("save_interval", 50))

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.out_dir = os.path.join(root, config.get("grpo_output_dir", "grpo_ckpt"))
        os.makedirs(self.out_dir, exist_ok=True)
        sft_dir = config.get("sft_output_dir")

        self.tokenizer = AutoTokenizer.from_pretrained(
            sft_dir, local_files_only=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.padding_side = "right"

        self.policy = AutoModelForCausalLM.from_pretrained(
            sft_dir, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="eager"
        ).to(device)

        self.reference = AutoModelForCausalLM.from_pretrained(
            sft_dir, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="eager"
        ).to(device)
        self.reference.eval()

        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=self.lr)
        self.scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=int(config.get("rl_warmup_steps", 200)),
            num_training_steps=self.max_updates
        )

        self.loader, self.gold_answers = None, []
        self.metrics = defaultdict(list)

    def prepare_data(self):
        self.logger.info(f"Loading dataset {self.cfg['dataset_name']}")
        splits = load_split_dataset_from_hf(
            dataset_name=self.cfg["dataset_name"],
            seed=self.seed,
            sft_ratio=self.cfg["split_ratios"]["sft"],
            rl_ratio=self.cfg["split_ratios"]["rl"],
            val_ratio=self.cfg["split_ratios"]["val"]
        )
        rl_raw = splits["rl_train"]
        self.gold_answers = [extract_gold_answer(x["answer"].strip()) for x in rl_raw]

        def _preproc(ex):
            prompt = format_cot_prompt(ex["question"])
            tok = self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            tok = {k: v.squeeze(0) for k, v in tok.items()}
            tok["gold"] = extract_gold_answer(ex["answer"].strip())
            return tok

        rl_ds = rl_raw.map(_preproc, remove_columns=rl_raw.column_names)
        rl_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        self.loader = get_dataloader(
            rl_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=int(self.cfg.get("num_workers", 4))
        )
        self.logger.info(f"{len(rl_ds)} RL prompts loaded.")

    @torch.no_grad()
    def _log_prob(self, model, seqs):

        # add attention mask
        attention_mask = (seqs != self.tokenizer.pad_token_id).long()
        
        # compute log probabilities for sequences
        lbl = seqs.clone()
        lbl[lbl == self.tokenizer.pad_token_id] = -100
        
        with torch.autocast(device_type=self.device.type, enabled=False):
            out = model(input_ids=seqs, attention_mask=attention_mask, labels=lbl)
        
        valid_len = (lbl != -100).sum(dim=1)
        lp = -out.loss * valid_len
        
        return torch.nan_to_num(lp, nan=-1e4, posinf=1e4, neginf=-1e4)

    def _sample_batch(self, batch):
        # generate sequences from policy model
        inp = {
            k: v.unsqueeze(1).to(self.device) for k, v in batch.items()
        }
        
        # Ensure input_ids has the right shape
        if inp["input_ids"].dim() == 2:
            inp["input_ids"] = inp["input_ids"].unsqueeze(0)
        
        # Add attention mask if not present
        if "attention_mask" not in inp:
            inp["attention_mask"] = torch.ones_like(inp["input_ids"])
        
        self.policy.eval()
        
        with torch.no_grad():
            seqs = self.policy.generate(
                **inp,
                do_sample=False,
                max_new_tokens=self.gen_len,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=False
            )
        
        self.policy.train()

        return seqs

    def _compute_reward(self, seqs):
        # compute final answer reward
        final_r = []
        for s, gold in zip(seqs, self.gold_answers):
            pred = extract_predicted_answer(
                self.tokenizer.decode(s, skip_special_tokens=True)
            )
            final_r.append(1.0 if pred == gold else 0.0)
        final_r = torch.tensor(final_r, device=self.device)

        # compute verifier bonus
        bonus = []
        for s in seqs:
            cot = self.tokenizer.decode(s, skip_special_tokens=True).split("A:")[-1]
            steps = [t.strip() for t in cot.split('.') if t.strip()]
            enc = self.verifier_tok(
                steps, padding=True, truncation=True, max_length=64,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.verifier(**enc).logits
                probs  = torch.softmax(logits, -1)[:, 1]
            bonus.append(probs.mean())
        bonus = torch.stack(bonus).to(self.device) * self.alpha
        return final_r + bonus

    def train(self):
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        vdir = self.cfg["verifier_dir"]
        self.verifier_tok = DistilBertTokenizerFast.from_pretrained(vdir, local_files_only=True)
        self.verifier = DistilBertForSequenceClassification.from_pretrained(vdir, local_files_only=True).to(self.device)
        self.verifier.eval()

        self.prepare_data()
        data_iter = iter(self.loader)

        # Main training loop
        for upd in range(self.max_updates):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.loader)
                batch = next(data_iter)
            
            batch.pop("attention_mask", None)
            batch_gold = batch.pop("gold", None) # maybe for later

            # sample and compute log probabilities
            seqs         = self._sample_batch(batch)
            old_logp_pol = self._log_prob(self.policy,     seqs)
            logp_ref     = self._log_prob(self.reference,  seqs)

            # compute reward and kl divergence
            R = self._compute_reward(seqs)
            KL = torch.nan_to_num(old_logp_pol - logp_ref, nan=0.0)
            reward = R - self.beta * KL

            # compute importance weights
            w = torch.exp(reward).clamp(1e-4, 1e4)
            w = w / w.mean().clamp(min=1e-8)

            # compute policy loss
            labels = seqs.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            attention_mask = (seqs != self.tokenizer.pad_token_id).long()

            out = self.policy(input_ids=seqs, attention_mask=attention_mask, labels=labels)
            
            logp = -F.cross_entropy(
                out.logits.transpose(1, 2), labels,
                reduction="none", ignore_index=-100
            ).sum(1)

            loss = -(w * logp).mean()

            # skip batch if loss is invalid
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning("NaN/Inf loss – batch skipped"); continue

            # update model parameters
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step(); self.scheduler.step()

            # log metrics and save checkpoints
            self.metrics["total_reward"].append(reward.mean().item())
            if upd % 10 == 0:
                self.logger.info(
                    f"Upd {upd}/{self.max_updates} • "
                    f"R̄={R.mean():.3f}  KL={KL.mean():.3f}  w̄={w.mean():.2f}"
                )
            if upd % self.save_every == 0:
                ck = os.path.join(self.out_dir, f"update_{upd}")
                self.policy.save_pretrained(ck, safe_serialization=False)
                self.tokenizer.save_pretrained(ck, safe_serialization=False)

        # save final model and evaluate
        self.policy.save_pretrained(self.out_dir, safe_serialization=False)
        self.tokenizer.save_pretrained(self.out_dir, safe_serialization=False)

        evaluator = ModelEvaluator(
            config=self.cfg, device=self.device,
            model=self.policy, tokenizer=self.tokenizer
        )
        res = evaluator.evaluate(num_samples=0)
        self.logger.info(f"GRPO finished • Test-Acc {res['test_accuracy']:.2f}%")

        self._plot()

    def _plot(self):
        
        # plot training metrics
        if not self.metrics["total_reward"]:
            return
        plt.figure(); plt.plot(self.metrics["total_reward"])
        plt.title("Average reward"); plt.xlabel("update"); plt.ylabel("R̄")
        plt.savefig(os.path.join(self.out_dir, "reward_curve.png"))
        plt.close()