import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
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

# Class-wise weights (example values from training phase, replace with actual values)
weights = {
    'inceptionv3': [0.9156, 0.9139, 0.5801],  
    'wideresnet': [0.5542, 0.9750, 0.4943], 
    'densenet': [0.6872, 0.9256, 0.5331]   
}

# Load the dataloaders to get class names
dataloaders, dataset_sizes, class_names = get_dataloaders()

# Function to preprocess a single image
def load_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to get predictions for a single image
def get_single_image_prediction(models_dict, image, device):
    image = image.to(device)
    probs_dict = {}
    
    with torch.no_grad():
        for model_name, model in models_dict.items():
            model = model.to(device)
            outputs = model(image)
            if model_name == 'inceptionv3' and isinstance(outputs, models.inception.InceptionOutputs):
                outputs = outputs.logits  # Extract the primary output for inception_v3
            probs = torch.nn.functional.softmax(outputs, dim=1)
            probs_dict[model_name] = probs.cpu().numpy()
    
    return probs_dict

# Function to calculate weighted ensemble probabilities for a single image
def get_weighted_ensemble_probs_single_image(probs_dict, weights_dict, num_classes):
    ensemble_probs = np.zeros((1, num_classes))  # Only one image
    
    for model_name in probs_dict.keys():
        for i in range(num_classes):
            ensemble_probs[:, i] += weights_dict[model_name][i] * probs_dict[model_name][:, i]
    
    ensemble_probs /= len(probs_dict)
    return ensemble_probs

# Path to the folder containing images you want to predict
image_folder_path = '/media/App1/shambhavi/udayan_lab_patches/H4U04009C'  # Replace with your folder path

# List to store results
results = []

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Iterate over all images in the folder with a progress bar
for image_name in tqdm(os.listdir(image_folder_path), desc="Processing images"):
    image_path = os.path.join(image_folder_path, image_name)
    if os.path.isfile(image_path) and image_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
        # Load the image
        image = load_image(image_path)

        # Get probabilities for the single image from each model
        probs_dict = get_single_image_prediction(models_dict, image, device)

        # Calculate weighted ensemble probabilities for the single image
        ensemble_probs = get_weighted_ensemble_probs_single_image(probs_dict, weights, num_classes)

        # Get the final prediction
        ensemble_preds = np.argmax(ensemble_probs, axis=1)

        # Append result to the list
        results.append([image_path, class_names[ensemble_preds[0]]])

# Create a DataFrame and save to CSV
df = pd.DataFrame(results, columns=['Image_Path', 'Ensemble_Prediction'])
df.to_csv('ensemble_predictions.csv', index=False)

print("Predictions saved to ensemble_predictions.csv")
