import torch
from torchvision import models
from dataset import get_dataloaders
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score
import numpy as np

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

# Get dataloaders
dataloaders, dataset_sizes, class_names = get_dataloaders()

# Function to get evaluation metrics
def get_metrics(model, dataloader, device):
    model = model.to(device)
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_probs), np.array(all_labels)

# Calculate evaluation metrics for each model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
metrics = {}

for model_name, model in models_dict.items():
    probs, labels = get_metrics(model, dataloaders['validate'], device)
    metrics[model_name] = {
        'probs': probs,
        'labels': labels
    }

# weights from test set
weights = {
    'inceptionv3': [1.6425,1.6905,1.6251],
    'wideresnet': [1.7503,1.7446,1.7512],
    'densenet': [1.7491,1.7628,1.7159]
}
#weight from validation set
#weights = {
#    'inceptionv3': [1.4117, 1.6358, 1.6783, 1.6643],
#    'wideresnet': [1.5870 ,1.7138, 1.7632, 1.7634],
#    'densenet': [1.6574, 1.7103, 1.7700, 1.7893]
#}

# Function to calculate weighted average ensemble probabilities
def get_weighted_ensemble_probs(probs_dict, weights_dict, num_classes):
    num_images = probs_dict['inceptionv3'].shape[0]
    ensemble_probs = np.zeros((num_images, num_classes))
    
    for model_name in probs_dict.keys():
        for i in range(num_classes):
            ensemble_probs[:, i] += weights_dict[model_name][i] * probs_dict[model_name][:, i]
    
    ensemble_probs /= len(probs_dict)
    return ensemble_probs

# Calculate weighted ensemble probabilities
probs_dict = {model_name: metrics[model_name]['probs'] for model_name in models_dict.keys()}
ensemble_probs = get_weighted_ensemble_probs(probs_dict, weights, num_classes)

# Get the final predictions
ensemble_preds = np.argmax(ensemble_probs, axis=1)
true_labels = metrics['inceptionv3']['labels']  # assuming all models use the same validation set

# Compute the confusion matrix and evaluation metrics
conf_matrix = confusion_matrix(true_labels, ensemble_preds)
accuracy = accuracy_score(true_labels, ensemble_preds)
f1 = f1_score(true_labels, ensemble_preds, average=None)
recall = recall_score(true_labels, ensemble_preds, average=None)
precision = precision_score(true_labels, ensemble_preds, average=None)

# Print the results
print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
for i, class_name in enumerate(class_names):
    print(f"Class {class_name} - F1 Score: {f1[i]:.4f}, Recall: {recall[i]:.4f}, Precision: {precision[i]:.4f}")

# Calculate and print AUC for each class
auc_scores = []
for i in range(num_classes):
    true_binary = (true_labels == i).astype(int)
    auc = roc_auc_score(true_binary, ensemble_probs[:, i])
    auc_scores.append(auc)
    print(f"AUC for class {class_names[i]}: {auc:.4f}")

mean_auc = np.mean(auc_scores)
print(f"Mean AUC: {mean_auc:.4f}")

# Calculate and print micro-averaged AUC-ROC
true_binary_micro = np.zeros((true_labels.size, num_classes))
for i in range(num_classes):
    true_binary_micro[:, i] = (true_labels == i).astype(int)

auc_micro = roc_auc_score(true_binary_micro, ensemble_probs, average="micro")
print(f"Micro-Averaged AUC-ROC: {auc_micro:.4f}")
