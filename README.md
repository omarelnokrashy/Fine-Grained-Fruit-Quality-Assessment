Based on the project report you provided, here is a well-formatted README file ready to be added to your GitHub repository. It includes clear headings, a project summary, details on the dataset, descriptions of the models, and a comparison of the results.

***

# [cite_start]Fine-Grained Fruit Quality Assessment [cite: 2]

## Project Overview

[cite_start]This project is a deep learning-based image classification system designed to perform fine-grained classification of fruit quality[cite: 7]. [cite_start]The model analyzes image features to assess and classify the ripeness levels and spoilage signs of bananas and tomatoes[cite: 8]. [cite_start]The classification system includes seven distinct categories: banana overripe, banana ripe, banana rotten, banana unripe, tomato fully ripened, tomato green, and tomato half-ripened[cite: 9].

[cite_start]This project was completed as part of a Neural Networks course[cite: 3].

## Dataset

[cite_start]The dataset consists of labeled food images representing different stages of ripeness and quality for bananas and tomatoes[cite: 11]. [cite_start]Each image is assigned to one of seven classes[cite: 12].

The class names and their corresponding labels are:

| Class Name | Label |
| :--- | :--- |
| banana_overripe | 0 |
| banana_ripe | 1 |
| banana_rotten | 2 |
| banana_unripe | 3 |
| tomato_fully_ripened | 4 |
| tomato_green | 5 |
| tomato_half_ripened | 6 |

### Addressing Class Imbalance

[cite_start]The initial dataset had a significant class imbalance, with banana-related classes having over 1500 images each, while some tomato classes had fewer than 200[cite: 15]. [cite_start]To address this, targeted data augmentation was applied to increase the number of samples in the minority classes[cite: 19]. [cite_start]Augmentation methods included horizontal/vertical flips, rotations, brightness adjustments, and zooming[cite: 24]. [cite_start]In addition, class weighting was used during model training to penalize errors from minority classes more heavily[cite: 28, 29].

## Model Architectures

Three different deep learning models were developed and compared for this project.

### 1. Hybrid CNN–Transformer Model

[cite_start]This hybrid model combines a Convolutional Neural Network (CNN) for feature extraction with a Vision Transformer (ViT) for classification[cite: 37]. [cite_start]The CNN captures local spatial features, and these features are then reshaped into a sequence of patches for the transformer[cite: 38, 39]. [cite_start]The Vision Transformer uses multi-head self-attention to capture global relationships[cite: 40].

* [cite_start]**Optimizer**: AdamW [cite: 44]
* [cite_start]**Loss Function**: Categorical Cross-Entropy [cite: 45]
* [cite_start]**Regularization**: Dropout, Early Stopping, Learning Rate Scheduling [cite: 46]
* [cite_start]**Metrics**: Accuracy, Precision, Recall, F1-Score [cite: 47]

### 2. AlexNet Model

[cite_start]The AlexNet model was adapted for this classification task[cite: 53, 54]. [cite_start]The architecture uses five convolutional layers and three fully connected layers to progressively extract hierarchical features[cite: 55, 56]. [cite_start]The model incorporates ReLU activation functions, dropout regularization, and max pooling to prevent overfitting[cite: 57].

* [cite_start]**Input Size**: $224 \times 224 \times 3$ (RGB image) [cite: 60]
* [cite_start]**Optimizer**: AdamW [cite: 73]
* [cite_start]**Loss Function**: Categorical Cross-Entropy [cite: 74]
* [cite_start]**Regularization**: Dropout, Early Stopping, Learning Rate Scheduling [cite: 75]
* [cite_start]**Metrics**: Accuracy, Precision, Recall, F1-Score [cite: 76]

### 3. Inception Model

[cite_start]This architecture is a simplified version of the Inception network, which is known for its multi-path feature extraction[cite: 81, 83]. [cite_start]A custom Inception module with three parallel "towers" was designed to process the input through multiple receptive fields and pooling paths[cite: 85].

* [cite_start]**Input Size**: $224 \times 224 \times 3$ (RGB image) [cite: 87]
* [cite_start]**Optimizer**: AdamW [cite: 97]
* [cite_start]**Loss Function**: Categorical Cross-Entropy [cite: 98]
* [cite_start]**Regularization**: Dropout, Early Stopping, Learning Rate Scheduling [cite: 99]
* [cite_start]**Metrics**: Accuracy, Precision, Recall, F1-Score [cite: 100]

## Results and Comparison

[cite_start]All three models were trained and evaluated under similar conditions to ensure a fair comparison[cite: 108]. [cite_start]The Hybrid CNN–Transformer model achieved the highest accuracy and showed robust generalization, likely due to its ability to capture both local and global features[cite: 112].

| Model | Training Accuracy | Validation Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- |
| CNN–Transformer | 0.9806 | 0.95 | 0.95 |
| AlexNet | 0.9204 | 0.91 | 0.93 |
| Inception | 0.7457 | 0.77 | 0.78 |

[cite_start]The AlexNet model performed well in early training but showed signs of overfitting, while the Inception model offered a good balance of accuracy and computational efficiency but did not achieve high accuracy[cite: 113, 114].
