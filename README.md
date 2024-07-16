# HER2 Classification from H&E Slides

This project demonstrates how to classify HER2 from H&E slides using the BCI dataset from the paper "A Novel Decision Level Class-Wise Ensemble Method in Deep Learning for Automatic Multi-Class Classification of HER2 Breast Cancer Hematoxylin-Eosin Images". The project utilizes a weighted ensemble deep learning method and achieves an AUC of 0.96. The original paper uses GoogLeNet, which has been replaced with InceptionV3 to stabilize training and overcome overfitting.

![image_matrix](https://github.com/user-attachments/assets/47e136f5-10af-41e4-b2c3-2307fe4bd565)

## Steps

### Step 1: Install the BCI Dataset
Download and install the BCI dataset from [this link](https://github.com/bupt-ai-cz/BCI).

### Step 2: Preprocess the Dataset
Convert all the H&E patches in the dataset to 512x512, and ensure the tissue threshold is set to 80%.

### Step 3: Train the Models
Run `train.py`. This will train the models and save all three models.

### Step 4: Calculate the Weight Matrix
Run `weight.py` to get the weight matrix.

### Step 5: Get Accuracy and AUC Scores
Add the weight matrix obtained in Step 4 to `get_metrics.py` and run it to get the Accuracy and AUC scores.

## Logs
All logs related to the project can be found at [this link](https://wandb.ai/shash100/HER2_pateel_3_DA_inceptionv3_with_stain?nw=nwusershash05).


