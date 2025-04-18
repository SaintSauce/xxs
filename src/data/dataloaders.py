from torch.utils.data import DataLoader

def get_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int
) -> DataLoader:
    """

    return pytorch dataloader
    
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )