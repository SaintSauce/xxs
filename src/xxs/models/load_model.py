import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class HFModelLoader:
    """ big mac class for the model """
    
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device     = device

    def load(self):
        """ return the model and tokenizer """

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        # Model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )

        model.to(self.device)

        return model, tokenizer

# loader = HFModelLoader("Qwen/Qwen2.5-0.5B", device)
# model, tokenizer = loader.load()