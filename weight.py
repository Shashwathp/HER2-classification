import torch
from torchvision import models
from dataset import get_dataloaders
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np
from tqdm import tqdm

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

# Hyperparameters
num_classes = 3

# Load models
models_dict = {
    'inceptionv3': load_model('inceptionv3', num_classes, 'inceptionv3.pkl'),
    'wideresnet': load_model('wideresnet', num_classes, 'wideresnet.pkl'),
    'densenet': load_model('densenet', num_classes, 'densenet.pkl'),
}

# Get dataloaders
dataloaders, dataset_sizes, class_names = get_dataloaders()

# Function to get evaluation metrics
def get_metrics(model, dataloader, device):
    model = model.to(device)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            if model_name == 'inceptionv3' and isinstance(outputs, models.inception.InceptionOutputs):
                outputs = outputs.logits  # Extract the primary output for inception_v3
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    precision = precision_score(all_labels, all_preds, average=None)
    
    return accuracy, f1, recall, precision

# Calculate evaluation metrics for each model 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
metrics = {}

for model_name, model in models_dict.items():
    accuracy, f1, recall, precision = get_metrics(model, dataloaders['validate'], device)
    metrics[model_name] = {
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'precision': precision
    }
 
# Weight function
def weight_function(x):
    alpha = 5
    beta = 2.5
    return alpha / (1 + np.exp(-2 * x)) - beta

# Calculate weights
weight_matrix = {}

for model_name, metric in metrics.items():
    accuracy = metric['accuracy']
    f1 = metric['f1']
    recall = metric['recall']
    precision = metric['precision']
    
    weights = []
    for i in range(num_classes):
        acc_weight = weight_function(accuracy)
        f1_weight = weight_function(f1[i])
        recall_weight = weight_function(recall[i])
        precision_weight = weight_function(precision[i])
        
        class_weight = (acc_weight + f1_weight + recall_weight + precision_weight) / 4
        weights.append(class_weight)
    
    weight_matrix[model_name] = weights

# Display the weight matrix
for model_name, weights in weight_matrix.items():
    print(f"Model: {model_name}")
    for i, weight in enumerate(weights):
        print(f"Class {class_names[i]}: Weight {weight:.4f}")
