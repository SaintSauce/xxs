import os
import math
import torch
import matplotlib.pyplot as plt
from transformers import (
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from xxs.utils.data import (
    load_split_dataset_from_hf,
    format_cot_prompt,
    combine_prompt_answer,
    get_dataloader,
    extract_predicted_answer,
    extract_gold_answer
)
from xxs.evaluation.eval_model import ModelEvaluator
from xxs.utils.config import ConfigLoader
from xxs.models.load_model import HFModelLoader

class SFTTrainer:
    """ supervised fine-tuning trainer for CoT data, with plotting """
    
    def __init__(self, config: ConfigLoader, device: torch.device):
        
        # load config fields
        self.config         = config

        # life is long
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

        raw_output_dir      = config.get("sft_output_dir", "sft_checkpoint")
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
        self.val_loss_loader= None
        self.val_answers    = None
        self.model          = None

        # metrics history
        self.epochs         = []
        self.train_loss     = []
        self.val_loss       = []
        self.val_acc        = []

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

        # splits
        splits = load_split_dataset_from_hf(
            dataset_name=self.dataset_name,
            seed=self.seed,
            sft_ratio=self.ratios["sft"],
            rl_ratio=self.ratios["rl"],
            val_ratio=self.ratios["val"]
        )
        sft_raw = splits["sft_train"]
        val_raw = splits["val"]

        # sft loader
        def prep_sft(ex):
            prompt = format_cot_prompt(ex["question"])
            answer = ex["answer"].strip()
            
            p_len  = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            
            full   = combine_prompt_answer(prompt, answer, self.tokenizer.eos_token)
            
            tk     = self.tokenizer(
                full, 
                truncation=True, 
                padding="max_length",
                max_length=self.max_length
            )
            
            labels = tk["input_ids"].copy()
            
            # mask out prompt tokens
            labels[:p_len] = [-100] * p_len

            tk["labels"] = labels
            
            return tk

        sft_ds = sft_raw.map(prep_sft, remove_columns=sft_raw.column_names)
        sft_ds.set_format(
            type="torch",
            columns=["input_ids","attention_mask","labels"]
        )
        
        self.loader = get_dataloader(
            sft_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=True, drop_last=True
        )

        # val_loss loader (for masked‐loss)
        val_loss_ds = val_raw.map(prep_sft, remove_columns=val_raw.column_names)
        val_loss_ds.set_format(
            type="torch",
            columns=["input_ids","attention_mask","labels"]
        )
        
        self.val_loss_loader = get_dataloader(
            val_loss_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True, drop_last=False
        )

        # val_gen loader (for accuracy)
        def prep_val(ex):
            prompt = format_cot_prompt(ex["question"])
            return self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

        val_gen_ds = val_raw.map(prep_val, remove_columns=val_raw.column_names)
        val_gen_ds.set_format(type="torch",
                              columns=["input_ids","attention_mask"])
        self.val_loader = get_dataloader(
            val_gen_ds, batch_size=1,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True
        )

        #  gold answers for accuracy
        self.val_answers = [ex["answer"].strip() for ex in val_raw]

    @torch.no_grad()
    def evaluate(self):
        """ return (val_loss, val_accuracy) for current model """
        
        self.model.eval()

        # validation loss
        total_loss, count = 0.0, 0
        for batch in self.val_loss_loader:
            b = {
                k: v.to(self.device) for k,v in batch.items()
            }
            out = self.model(**b)
            total_loss += out.loss.item()
            count += 1
        avg_loss = total_loss / count

        # exact-match accuracy
        correct = 0

        for i, batch in enumerate(self.val_loader):
            b = {k: v.squeeze(1).to(self.device) for k,v in batch.items()}
            out_ids   = self.model.generate(**b, max_new_tokens=128)
            txt       = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            pred_ans  = extract_predicted_answer(txt)
            gold_ans  = extract_gold_answer(self.val_answers[i])
            
            # exact match
            if pred_ans == gold_ans:
                correct += 1
        
        acc = correct / len(self.val_loader) * 100

        self.model.train()
        return avg_loss, acc

    def _save_plots(self):
        """ plot & save training+validation metrics """
        
        # loss curve
        plt.figure(figsize=(8,6))
        plt.plot(self.epochs, self.train_loss, label="Train Loss", marker="o")
        plt.plot(self.epochs, self.val_loss,   label="Val Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("SFT Loss Curve")
        plt.savefig(os.path.join(self.output_dir, "loss_curve.png"))
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
        """ run SFT with per-epoch validation and final plotting """
        
        optimizer = AdamW(self.model.parameters(),
                          lr=self.lr,
                          weight_decay=self.weight_decay)
        
        total_steps = math.ceil(len(self.loader)/self.grad_accum) * self.num_epochs
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        for epoch in range(1, self.num_epochs + 1):
            self.epochs.append(epoch)
            running_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(self.loader, start=1):
                b = {
                    k: v.to(self.device) for k,v in batch.items()
                }
                out = self.model(**b)
                loss = out.loss / self.grad_accum
                loss.backward()
                running_loss += loss.item()
                num_batches += 1

                if step % self.grad_accum == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # Calculate average loss for this epoch
            avg_epoch_loss = running_loss / num_batches
            self.train_loss.append(avg_epoch_loss)
            print(f"[Epoch {epoch}] Average loss: {avg_epoch_loss:.4f}")

            # checkpoint
            ckpt = os.path.join(self.output_dir, f"epoch{epoch}")
            self.model.save_pretrained(ckpt)
            self.tokenizer.save_pretrained(ckpt)
            print(f"Saved checkpoint: {ckpt}")

            # validation
            val_l, val_a = self.evaluate()
            print(f"Val Loss: {val_l:.4f} | Acc: {val_a:.2f}%")
            self.val_loss.append(val_l)
            self.val_acc.append(val_a)

            # optional verify
            if self._verify_save(ckpt):
                print(f"Verified checkpoint: {ckpt}")

        # final save
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # verify final save
        if self._verify_save(self.output_dir):
            print(f"Final model saved & verified in {self.output_dir}")

        evaluator = ModelEvaluator(
            config    = self.config,
            device    = self.device,
            model     = self.model,
            tokenizer = self.tokenizer
        )
        test_metrics = evaluator.evaluate(0)  # prints 5 examples

        print(
            f"Test Accuracy: {test_metrics['test_accuracy']:.2f}% "
            f"on {test_metrics['test_samples']} samples"
        )

        # finally, save plots
        self._save_plots()
        print(f"Saved loss_curve.png & accuracy_curve.png to {self.output_dir}")