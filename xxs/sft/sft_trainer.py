import os, math, torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdamW,
    get_linear_schedule_with_warmup
)
from xxs.utils.data import (
    load_split_dataset_from_hf,
    format_cot_prompt,
    combine_prompt_answer,
    get_dataloader
)
from xxs.utils.config import ConfigLoader
from xxs.models.load_model import HFModelLoader

class SFTTrainer:
    """ supervised fine-tuning trainer for CoT data """
    
    def __init__(
        self,
        config: ConfigLoader,
        device: torch.device
    ):
        
        # load config fields
        self.dataset_name = config.get("dataset_name")
        self.model_name   = config.get("model_name")
        self.ratios       = config.get("split_ratios")
        self.seed         = config.get("seed")
        self.max_length   = config.get("max_length")
        self.batch_size   = config.get("sft_batch_size")
        self.num_workers  = config.get("num_workers")
        self.lr           = config.get("learning_rate")
        self.weight_decay = config.get("weight_decay")
        self.num_epochs   = config.get("num_epochs")
        self.grad_accum   = config.get("grad_accum_steps")
        self.warmup_steps = config.get("warmup_steps")

        raw_output_dir    = config.get("sft_output_dir", "./sft_checkpoint")
        
        # Compute repo root
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        # Make output_dir always live under repo root
        self.output_dir   = os.path.join(repo_root, raw_output_dir)

        self.device       = device
        self.model_loader = HFModelLoader(self.model_name, self.device)

        # placeholders
        self.tokenizer    = None
        self.loader       = None
        self.model        = None

    def _verify_save(self, save_dir: str) -> bool:
        """ verify that model and tokenizer files are properly saved """

        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json"
        ]
        
        for file in required_files:
            file_path = os.path.join(save_dir, file)
            if not os.path.exists(file_path):
                print(f"Warning: Required file {file} not found in {save_dir}")
                return False
        return True

    def prepare_data(self):
        """ load and preprocess the dataset """
        
        # load splits
        splits = load_split_dataset_from_hf(
            dataset_name=self.dataset_name,
            splits=self.ratios,
            seed=self.seed
        )
        sft_raw = splits["sft"]

        # load tokenizer
        _, self.tokenizer = self.model_loader.load()

        # preprocess
        def prep_sft(sample):
            prompt = format_cot_prompt(sample["question"])
            answer = sample["answer"].strip()
            
            p_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            full = combine_prompt_answer(prompt, answer, self.tokenizer.eos_token)
            
            tk = self.tokenizer(
                full,
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )
            
            labels = tk["input_ids"].copy()
            labels[:p_len] = -100
            tk["labels"] = labels

            return tk
    
        ds = sft_raw.map(prep_sft, remove_columns=sft_raw.column_names)
        ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
        
        # get dataloader
        self.loader = get_dataloader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def train(self):
        """ train the model """

        self.model, self.tokenizer = self.model_loader.load()

        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        n_batches = len(self.loader)

        # use more batches for optimization
        total_steps = math.ceil(n_batches / self.grad_accum) * self.num_epochs
        
        # warmup + linear decay
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        # make output dir
        os.makedirs(self.output_dir, exist_ok=True)

        # train
        self.model.train()
        global_step = 0
        
        for epoch in range(1, self.num_epochs + 1):
            running_loss = 0.0
            
            # train loop
            for step, batch in enumerate(self.loader, start=1):
                
                # gradient accumulation
                batch = {
                    k: v.to(self.device) for k,v in batch.items()
                }

                out = self.model(**batch)
                loss = out.loss / self.grad_accum
                loss.backward()
                running_loss += loss.item()

                if step % self.grad_accum == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # logging
                    if global_step % 50 == 0:
                        print(f"[Epoch {epoch}] Step {global_step} | avg loss: {running_loss / 50:.4f}")
                        running_loss = 0.0

            # checkpoint
            ckpt = os.path.join(self.output_dir, f"epoch{epoch}")
            self.model.save_pretrained(ckpt)
            self.tokenizer.save_pretrained(ckpt)
            
            # Verify checkpoint save
            if self._verify_save(ckpt):
                print(f"Successfully saved and verified checkpoint: {ckpt}")
            else:
                print(f"Warning: Checkpoint verification failed for {ckpt}")

        # final save
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Verify final save
        if self._verify_save(self.output_dir):
            print(f"SFT complete. Model + tokenizer successfully saved and verified in {self.output_dir}")
        else:
            print(f"Warning: Final model save verification failed in {self.output_dir}")