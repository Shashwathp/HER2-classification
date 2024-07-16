import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(train_dir='/media/App1/Shashwath/HER2/HER2_pateel_3/dataset/train', validate_dir='/media/App1/Shashwath/HER2/HER2_pateel_3/dataset/validate', batch_size=20):
    # Image transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((299, 299)),  # Updated to 299x299
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Translation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validate': transforms.Compose([
            transforms.Resize((299, 299)),  # Updated to 299x299
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'validate': datasets.ImageFolder(validate_dir, data_transforms['validate']),
    }

    # Data loaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'validate': DataLoader(image_datasets['validate'], batch_size=batch_size, shuffle=False),
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validate']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names
