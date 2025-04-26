
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloader(train_dir: str,
                      test_dir: str,
                      transforms: transforms.Compose,
                      batch_size: int = 32,
                      num_workers: int = NUM_WORKERS):
    train_data = datasets.ImageFolder(root=train_dir, transform=transforms)
    test_data = datasets.ImageFolder(root=test_dir, transform=transforms)

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, test_dataloader, class_names
