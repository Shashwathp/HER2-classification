import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset import get_dataloaders  # Import the function from dataset.py
from tqdm import tqdm  # For the progress bar
import wandb
import copy

# Initialize W&B
wandb.login()

# Get the dataloaders
dataloaders, dataset_sizes, class_names = get_dataloaders()

# Load pre-trained models and modify the last layer
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "inceptionv3":
        model = models.inception_v3(pretrained=use_pretrained)
        model.aux_logits = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "wideresnet":
        model = models.wide_resnet50_2(pretrained=use_pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet":
        model = models.densenet201(pretrained=use_pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

# Hyperparameters
num_classes = 3
learning_rate = 0.001
num_epochs = 100
step_size = 20
gamma = 0.9  # 10% decay means we keep 90% of the learning rate
early_stopping_patience = 7

# Initialize models
models_dict = {
    'inceptionv3': initialize_model('inceptionv3', num_classes, feature_extract=True),
    'wideresnet': initialize_model('wideresnet', num_classes, feature_extract=True),
    'densenet': initialize_model('densenet', num_classes, feature_extract=True),
}

# Define loss function and optimizers
criterion = nn.CrossEntropyLoss()
optimizers = {
    'inceptionv3': optim.SGD(models_dict['inceptionv3'].parameters(), lr=learning_rate, momentum=0.9),
    'wideresnet': optim.SGD(models_dict['wideresnet'].parameters(), lr=learning_rate, momentum=0.9),
    'densenet': optim.SGD(models_dict['densenet'].parameters(), lr=learning_rate, momentum=0.9),
}

# Learning rate schedulers
schedulers = {
    'inceptionv3': optim.lr_scheduler.StepLR(optimizers['inceptionv3'], step_size=step_size, gamma=gamma),
    'wideresnet': optim.lr_scheduler.StepLR(optimizers['wideresnet'], step_size=step_size, gamma=gamma),
    'densenet': optim.lr_scheduler.StepLR(optimizers['densenet'], step_size=step_size, gamma=gamma),
}

# Training function
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, model_name):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    # Initialize W&B run
    wandb.init(project="HER2-plco-3class-dataaugmentation-inceptionv3", name=model_name, mode="offline")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validate']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if model_name == 'inceptionv3':
                        outputs = outputs.logits  # Extract the primary output for inception_v3

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Log metrics to W&B
            wandb.log({f'{phase}_loss': epoch_loss, f'{phase}_acc': epoch_acc, 'epoch': epoch+1})

            # Deep copy the model
            if phase == 'validate':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        print()

        # Check early stopping
        if epochs_no_improve == early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1}')
            
            # Load best model weights
            model.load_state_dict(best_model_wts)

            # Save the best model
            model_save_path = f'{model_name}_best.pkl'
            torch.save(model.state_dict(), model_save_path)
            print(f'Best model saved to {model_save_path}')

            break

    if epochs_no_improve < early_stopping_patience:
        print(f'Best val Acc: {best_acc:4f}')

        # Load best model weights
        model.load_state_dict(best_model_wts)

        # Save the best model
        model_save_path = f'{model_name}_best.pkl'
        torch.save(model.state_dict(), model_save_path)
        print(f'Best model saved to {model_save_path}')

    # Finish the W&B run
    wandb.finish()

    return model

# Fine-tune each model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

for model_name, model in models_dict.items():
    model = model.to(device)
    optimizer = optimizers[model_name]
    scheduler = schedulers[model_name]
    print(f'Training {model_name}...')
    trained_model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, model_name)
    
    # Save the trained model
    model_save_path = f'{model_name}_best.pkl'
    torch.save(trained_model.state_dict(), model_save_path)
    print(f'{model_name} saved to {model_save_path}')
