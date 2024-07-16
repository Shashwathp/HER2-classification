import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
from dataset import get_dataloaders

# Load the trained models
def load_model(model_name, num_classes, model_path):
    if model_name == "inceptionv3":
        model = models.inception_v3(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "wideresnet":
        model = models.wide_resnet50_2(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet":
        model = models.densenet201(pretrained=True)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Load the models
num_classes = 3
models_dict = {
    'inceptionv3': load_model('inceptionv3', num_classes, 'inceptionv3.pkl'),
    'wideresnet': load_model('wideresnet', num_classes, 'wideresnet.pkl'),
    'densenet': load_model('densenet', num_classes, 'densenet.pkl'),
}

weights = {
    'inceptionv3': [1.0436, 1.0648, 0.6282],  
    'wideresnet': [0.9854, 0.9033, 0.6660], 
    'densenet': [0.9229, 1.0323, 0.6294]   
}

# Load the dataloaders to get class names
dataloaders, dataset_sizes, class_names = get_dataloaders()

# Function to preprocess an image
def load_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to get predictions for a batch of images
def get_batch_predictions(models_dict, image_batch, device):
    image_batch = image_batch.to(device)
    probs_dict = {model_name: [] for model_name in models_dict.keys()}
    
    with torch.no_grad():
        for model_name, model in models_dict.items():
            model = model.to(device)
            outputs = model(image_batch)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            probs_dict[model_name] = probs.cpu().numpy()
    
    return probs_dict

# Function to calculate weighted ensemble probabilities for a batch of images
def get_weighted_ensemble_probs_batch(probs_dict, weights_dict, num_classes):
    num_images = next(iter(probs_dict.values())).shape[0]
    ensemble_probs = np.zeros((num_images, num_classes))
    
    for model_name in probs_dict.keys():
        for i in range(num_classes):
            ensemble_probs[:, i] += weights_dict[model_name][i] * probs_dict[model_name][:, i]
    
    # Normalize probabilities to ensure they sum to 1
    ensemble_probs /= ensemble_probs.sum(axis=1, keepdims=True)
    return ensemble_probs

# Main folder containing subfolders of images
main_folder = '/media/App1/Shashwath/HER2/HER_PLCO_4/dataset_test_format_3'  # Replace with your main folder path

# Get all subfolders
subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function to process images in batches
def process_images_in_batches(images, batch_size):
    num_images = len(images)
    ensemble_probs_list = []

    for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = images[start_idx:end_idx]

        # Get probabilities for the batch of images from each model
        probs_dict = get_batch_predictions(models_dict, batch_images, device)

        # Calculate weighted ensemble probabilities for the batch of images
        batch_ensemble_probs = get_weighted_ensemble_probs_batch(probs_dict, weights, num_classes)

        ensemble_probs_list.append(batch_ensemble_probs)

    return np.vstack(ensemble_probs_list)

# Prepare a DataFrame to save results
results = []

# Process each subfolder
batch_size = 100
for subfolder in subfolders:
    print(f"Processing subfolder: {subfolder}")
    image_paths = [os.path.join(subfolder, fname) for fname in os.listdir(subfolder) if fname.endswith(('jpg', 'jpeg', 'png'))]
    images = [load_image(image_path) for image_path in image_paths]
    images = torch.cat(images)
    
    # Process images in batches
    ensemble_probs = process_images_in_batches(images, batch_size)

    # Collect results
    for image_path, probs in zip(image_paths, ensemble_probs):
        formatted_probs = [f"{prob:.4f}" for prob in probs]
        result = [image_path] + formatted_probs
        results.append(result)

# Create a DataFrame and save to CSV
df = pd.DataFrame(results, columns=['ImagePath', 'HER0_1', 'HER2', 'HER3'])
csv_path = 'her2_probabilities.csv'
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")
