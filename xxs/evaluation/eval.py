import torch
from typing import Dict, List, Tuple
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader

from xxs.utils.data import (
    load_split_dataset_from_hf,
    format_cot_prompt,
    get_dataloader
)
from xxs.utils.config import ConfigLoader
from xxs.models.load_model import HFModelLoader

class ModelEvaluator:
    """ evaluator for Hugging Face models on test sets """
    
    def __init__(self, config: ConfigLoader, device: torch.device):
        self.config = config
        self.device = device
        
        # Load model and tokenizer
        self.model_loader = HFModelLoader(
            model_name=config.get("model_name"),
            device=device
        )
        self.model, self.tokenizer = self.model_loader.load()
        
        # Get dataset parameters
        self.dataset_name = config.get("dataset_name")
        self.max_length = config.get("max_length")
        self.batch_size = config.get("batch_size")
        self.num_workers = config.get("num_workers")
        
        # Initialize data loaders
        self.test_loader = None
        self.test_answers = None
        
    def prepare_test_data(self):
        """ prepare test data loader and gold answers """
        
        # load test split
        splits = load_split_dataset_from_hf(
            dataset_name=self.dataset_name,
            seed=self.config.get("seed"),
            sft_ratio=0.0,  # We only need test split
            rl_ratio=0.0,
            val_ratio=0.0
        )
        test_raw = splits["test"]
        
        # Prepare test data
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
        
        self.test_loader = get_dataloader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # Store gold answers
        self.test_answers = [ex["answer"].strip() for ex in test_raw]
        
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """ evaluate model on test set and return metrics """
        
        if self.test_loader is None:
            self.prepare_test_data()
            
        self.model.eval()
        
        # Initialize metrics
        correct = 0
        total = len(self.test_loader)
        
        # Evaluate each batch
        for i, batch in enumerate(self.test_loader):
            # Move batch to device
            b = {k: v.to(self.device) for k, v in batch.items()}
            
            # Generate predictions
            out_ids = self.model.generate(
                **b,
                max_new_tokens=128,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode predictions
            pred_txt = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            pred_ans = pred_txt.strip().split()[-1]
            
            # Get gold answer
            gold_ans = self.test_answers[i].split()[-1]
            
            # Check exact match
            if pred_ans == gold_ans:
                correct += 1
                
        # Calculate accuracy
        accuracy = correct / total * 100
        
        return {
            "test_accuracy": accuracy,
            "test_samples": total
        }
        
def run_evaluation(config: ConfigLoader, device: torch.device):
    """ run evaluation on test set """
    
    evaluator = ModelEvaluator(config, device)
    metrics = evaluator.evaluate()
    
    print("\nEvaluation Results:")
    print(f"Test Accuracy: {metrics['test_accuracy']:.2f}%")
    print(f"Test Samples: {metrics['test_samples']}")
