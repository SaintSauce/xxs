import os, logging
from collections import defaultdict

import torch
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

        self.cfg = config
        self.device = device
        self.seed = int(config.get("seed"))
        self.max_len = int(config.get("max_length"))
        self.gen_len = int(config.get("gen_max_new_tokens", 128))

        self.batch_size = int(config.get("rl_batch_size"))
        self.lr = float(config.get("rl_learning_rate", 1e-5))
        self.epochs = int(config.get("grpo_epochs", 1))
        self.beta = float(config.get("kl_coef", 0.1))
        self.alpha = float(config.get("verifier_coef", 0.5))
        self.max_updates = int(config.get("max_updates", 1000))
        self.save_every = int(config.get("save_interval", 50))

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.out_dir = os.path.join(root, config.get("grpo_output_dir", "grpo_ckpt"))
        os.makedirs(self.out_dir, exist_ok=True)
        sft_dir = config.get("sft_output_dir")

        self.tokenizer = AutoTokenizer.from_pretrained(
            sft_dir, local_files_only=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.padding_side = "left"

        self.policy = AutoModelForCausalLM.from_pretrained(
            sft_dir, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        ).to(device)

        self.reference = AutoModelForCausalLM.from_pretrained(
            sft_dir, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
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
            tok["prompt_len"] = (tok["input_ids"] != self.tokenizer.pad_token_id).sum().item()
            return tok

        rl_ds = rl_raw.map(_preproc, remove_columns=rl_raw.column_names)
        rl_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "gold", "prompt_len"])

        self.loader = get_dataloader(
            rl_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=int(self.cfg.get("num_workers", 4))
        )
        
        self.logger.info(f"{len(rl_ds)} RL prompts loaded.")

    def _log_prob(self, model, seqs):
        """ compute log‐prob of seqs under model """
        
        # attention mask
        attn_mask = (seqs != self.tokenizer.pad_token_id).long()          # (B, L)
        B, L = seqs.size()
        
        # continuation mask
        idx = torch.arange(L, device=seqs.device).unsqueeze(0).expand(B, L)
        cont_mask = (idx >= (L - self.gen_len)) & (attn_mask.bool())
        
        # model forward under amp
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            logits = model(input_ids=seqs, attention_mask=attn_mask).logits  # (B, L, V)
        
        # log‐softmax + gather
        logp  = torch.nn.functional.log_softmax(logits, dim=-1)
        token_lp = logp.gather(-1, seqs.unsqueeze(-1)).squeeze(-1)        # (B, L)
        token_lp = token_lp * cont_mask.float()
        
        # sum over the continuation tokens
        return token_lp.sum(dim=1)                                         # (B,)

    def _sample_batch(self, batch, max_new_tokens: int):
        """ sample a batch of sequences from the policy model """
        
        # only keep the two tensors generate() needs
        inp = {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
        }

        self.policy.eval()
        
        with torch.no_grad():
            seqs = self.policy.generate(
                **inp,
                do_sample=True,
                top_k=40,
                top_p=0.8,
                temperature=0.8,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=False,
            )
        
        self.policy.train()
        
        return seqs

    def _compute_reward(self, seqs, gold_list, prompt_lens):
        
        # compute final answer reward
        final_r = []

        for s, gold, pl in zip(seqs, gold_list, prompt_lens):
            # only decode the newly generated portion
            gen_ids = s[pl:].tolist()
            gen_txt = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred   = extract_predicted_answer(gen_txt)
            final_r.append(1.0 if pred == gold else 0.0)
        final_r = torch.tensor(final_r, device=self.device)

        decoded = self.tokenizer.batch_decode(
            seqs, 
            skip_special_tokens=True
        )

        all_steps, counts = [], []
        
        for txt in decoded:
            # Split on "A:" if present, otherwise use the whole text
            if "A:" in txt:
                steps = [st.strip() for st in txt.split("A:")[-1].split('.') if st.strip()]
            else:
                steps = [txt.strip()]
            
            # Ensure we have at least one step
            if not steps:
                steps = [txt.strip()]
                
            counts.append(len(steps))
            all_steps.extend(steps)

        # If no steps were extracted, use the full decoded texts
        if not all_steps:
            all_steps = decoded
            counts = [1] * len(decoded)

        enc = self.verifier_tok(
            all_steps, padding=True, truncation=True, max_length=64,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.verifier(**enc).logits
            probs  = torch.softmax(logits, dim=-1)[:, 1]

        bonus = []
        idx = 0
        
        for c in counts:
            bonus.append(probs[idx:idx+c].mean())
            idx += c
        
        bonus = torch.stack(bonus) * self.alpha

        self.metrics["answer_acc"].append(final_r.mean().item())
        self.metrics["verifier_bonus"].append(bonus.mean().item())
        
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

            batch_gold = batch["gold"] # shape (B,)

            # draw K samples per prompt
            K = 4

            # remove gold from the batch dict before sampling
            batch_copy = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }

            # compute prompt length once (all inputs are padded to max_len)
            prompt_lens = batch["prompt_len"].to(self.device)
            target_len = prompt_lens.max().item() + self.gen_len
            
            self.policy.eval()

            with torch.no_grad():
                flat = self.policy.generate(
                    **batch_copy,
                    do_sample=True,
                    top_k=40, top_p=0.8, temperature=0.8,
                    max_new_tokens=self.gen_len,
                    num_return_sequences=K,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=False,
                )

            self.policy.train()
            
            # reshape from (B*K, L) -> (B, K, L)
            B = batch_copy["input_ids"].size(0)
            all_seqs = flat.view(B, K, -1)

            if upd % 100 == 0:
                
                # show up to 2 samples, K chains each, but only print the first chain per sample
                batch_size = batch_copy["input_ids"].size(0)
                
                for idx in range(min(2, batch_size)):
                    # how many tokens the prompt actually occupies
                    pl = prompt_lens[idx].item() if torch.is_tensor(prompt_lens) else prompt_lens[idx]

                    # take last pl tokens of the ORIGINAL input_ids as the question
                    orig_ids = batch_copy["input_ids"][idx]
                    prompt_ids = orig_ids[-pl:]
                    question_text = self.tokenizer.decode(
                        prompt_ids, skip_special_tokens=True
                    )

                    # take exactly the last gen_len tokens of the generated sequence
                    generated_ids = all_seqs[idx, 0, -self.gen_len:]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    pred_ans = extract_predicted_answer(generated_text)
                    gold_ans = batch_gold[idx]
                    
                    self.logger.info(f"Sample {idx+1}:")
                    self.logger.info(f"  Q: {question_text}")
                    self.logger.info(f"  Generated: {generated_text}")
                    self.logger.info(f"  Predicted Answer: {pred_ans}")
                    self.logger.info(f"  Gold Answer:     {gold_ans}")
                    self.logger.info("-" * 40)
            
            padded = []

            for seq in all_seqs:
                
                # Ensure sequence is exactly target_len by truncating if too long
                if seq.size(1) > target_len:
                    seq = seq[:, :target_len]
                
                pad_amount = target_len - seq.size(1)
            
                if pad_amount > 0:
                    # pad on the right up to target_len
                    seq = torch.nn.functional.pad(
                        seq,
                        (0, pad_amount), # (pad_left, pad_right)
                        value=self.tokenizer.pad_token_id
                    )
            
                padded.append(seq)
            
            all_seqs = torch.stack(padded, dim=0)              # now shape (B, K, max_len)

            # expand each gold answer K times so we have B*K gold labels
            flat_gold = []

            for g in batch_gold:
                flat_gold.extend([g] * K) # now len(flat_gold) == B*K
            
            # flatten B×K for log-prob calls
            flat_seqs = all_seqs.view(-1, all_seqs.size(-1)) # (B*K, L)
            flat_prompt_lens = prompt_lens.repeat_interleave(K).to(self.device)
            
            # score old/ref in small chunks to avoid OOM
            flat_old_chunks, flat_ref_chunks = [], []
            chunk_size = 32
            
            for i in range(0, flat_seqs.size(0), chunk_size):
                chunk = flat_seqs[i : i + chunk_size]
            
                pl_slice = flat_prompt_lens[i : i + chunk_size]

                with torch.no_grad():
                    flat_old_chunks.append(self._log_prob(self.policy, chunk))
                    flat_ref_chunks.append(self._log_prob(self.reference, chunk))
            
            flat_old_lp = torch.cat(flat_old_chunks, dim=0)
            flat_ref_lp = torch.cat(flat_ref_chunks, dim=0)

            old_lp = flat_old_lp.view(-1, K) # (B, K)
            ref_lp = flat_ref_lp.view(-1, K)

            flat_R = self._compute_reward(flat_seqs, flat_gold, flat_prompt_lens) # (B*K,)
            R = flat_R.view(-1, K) # (B, K)

            self.metrics["total_reward"].append(R.mean().item())

            # compute true group-relative weights
            KL = (old_lp - ref_lp).clamp(min=0) # (B, K)
            raw = R - self.beta * KL # (B, K)
            w = torch.softmax(raw / self.alpha, dim=1) # normalize across K

            kl_penalty = KL.mean().item()
            self.metrics["kl_penalty"].append(kl_penalty)

            # compute new log-probs under current policy
            # do the forward in 16-sequence chunks to save peak memory
            new_lp_chunks = []
            chunk_size = 32
            
            for i in range(0, flat_seqs.size(0), chunk_size):
                chunk = flat_seqs[i : i + chunk_size]
                new_lp_chunks.append(self._log_prob(self.policy, chunk))
            
            flat_new_lp = torch.cat(new_lp_chunks, dim=0)

            new_lp = flat_new_lp.view(-1, K) # (B, K)

            # GRPO loss: weighted sum over group
            loss = - (w * new_lp).sum(dim=1).mean()
            self.metrics["surrogate_loss"].append(loss.item())

            # skip batch if loss is invalid
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning("NaN/Inf loss – batch skipped"); continue

            # update model parameters
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()
            self.scheduler.step()

            if upd % 10 == 0:
                self.logger.info(
                    f"Upd {upd}/{self.max_updates} - "
                    f"Total Reward: {R.mean():.3f} | "
                    f"Answer Acc: {self.metrics['answer_acc'][-1]:.3f} | "
                    f"Verifier Bonus: {self.metrics['verifier_bonus'][-1]:.3f} | "
                    f"KL Penalty: {kl_penalty:.3f}"
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
        updates = range(len(self.metrics["total_reward"]))
        plt.figure(figsize=(8,5))
        plt.plot(updates, self.metrics["total_reward"],    label="Total Reward")
        plt.plot(updates, self.metrics["answer_acc"],      label="Answer Acc")
        plt.plot(updates, self.metrics["verifier_bonus"],  label="Verifier Bonus")
        plt.plot(updates, self.metrics["kl_penalty"],      label="KL Penalty")
        plt.xlabel("Update"); plt.ylabel("Value"); plt.title("Reward Components")
        plt.legend()
        plt.savefig(os.path.join(self.out_dir, "reward_components.png"))
        plt.close()