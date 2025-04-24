import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer
from xxs.utils.config import ConfigLoader
from xxs.models.load_model import HFModelLoader

from xxs.utils.data import format_cot_prompt, extract_predicted_answer, extract_gold_answer

class ModelEvaluator:
    """ model evaluator """

    def __init__(
        self,
        config: ConfigLoader,
        device: torch.device,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        ckpt_dir: Optional[str] = None
    ):
        self.config = config
        self.device = device

        self.dataset_name = config.get("dataset_name")
        self.max_length   = int(config.get("max_length"))
        self.batch_size   = int(config.get("batch_size"))
        self.num_workers  = int(config.get("num_workers"))

        self.test_loader  = None
        self.test_questions = []
        self.test_answers = []

        if model is not None and tokenizer is not None:
            # use the in‐memory SFT model
            self.model, self.tokenizer = model, tokenizer
        elif ckpt_dir is not None:
            # load from disk
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(ckpt_dir).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        else:
            # original behavior
            self.model_loader = HFModelLoader(
                model_name=config.get("model_name_of_model_to_evaluate"),
                device=device
            )
            self.model, self.tokenizer = self.model_loader.load()

    def prepare_test_data(self):

        # load test split directly
        ds = load_dataset(self.dataset_name, "main")
        test_raw = ds["test"]

        self.test_questions = [ex["question"].strip() for ex in test_raw]
        self.test_answers   = [ex["answer"].strip() for ex in test_raw]

        # tokenization for generation
        def prep_test(ex):
            prompt = format_cot_prompt(ex["question"])
            return self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

        test_ds = test_raw.map(prep_test, remove_columns=test_raw.column_names)
        test_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        # batch_size=1 so i indexes samples
        self.test_loader  = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    @torch.no_grad()
    def evaluate(self, num_samples: int = 5) -> Dict[str, float]:

        # prepare test data
        if self.test_loader is None:
            self.prepare_test_data()

        self.model.eval()

        correct, total = 0, len(self.test_answers)

        samples: List[tuple] = []

        for i, batch in enumerate(self.test_loader):
            
            # move to device
            b = {
              k: v.squeeze(1).to(self.device) for k,v in batch.items()
            }
            
            # generate
            out = self.model.generate(
                **b,
                max_new_tokens=128,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # decode
            text  = self.tokenizer.decode(out[0], skip_special_tokens=True)
            pred  = extract_predicted_answer(text)
            gold  = extract_gold_answer(self.test_answers[i])

            # exact match
            if pred == gold:
                correct += 1

            if len(samples) < num_samples:
                samples.append((self.test_questions[i], text, self.test_answers[i]))

        if samples:
            print(f"\n--- Showing {len(samples)} sample generations ---")
            for j, (q, gen, gold) in enumerate(samples, 1):
                print(f"\n[{j}] Q: {q}\nGenerated:\n{gen}\nGold Answer: {gold}")

        # accuracy
        acc = float(correct) / float(total) * 100
        print(f"Test Accuracy: {acc:.2f}% on {total} samples")

        return {
            "test_accuracy": acc, 
            "test_samples": total
        }