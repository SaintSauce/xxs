if __name__ == "__main__":
    from src.models.load_model import HFModelLoader
    from src.data.load_dataset import HFDatasetLoader
    from src.utils.utils import setup_logging

    # Setup logging
    logger = setup_logging()

    # Load model
    model_name = "Qwen/Qwen2.5-0.5B"
    device = "cuda:0"
    model_loader = HFModelLoader(model_name, device)
    model, tokenizer = model_loader.load()

    # Load dataset
    dataset_name = "Qwen/Qwen2.5-0.5B"
    dataset_loader = HFDatasetLoader(dataset_name, tokenizer)
    dataset = dataset_loader.load()