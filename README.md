Fine-Grained Fruit Quality Assessment
Project Overview
This project focuses on developing a deep learning-based image classification system for fine-grained fruit quality assessment. The system is designed to classify the ripeness levels and spoilage signs of bananas and tomatoes by analyzing image features. The classification is performed across seven distinct categories: banana overripe, banana ripe, banana rotten, banana unripe, tomato fully ripened, tomato green, and tomato half-ripened.

Dataset
The project uses a dataset of labeled food images representing various stages of ripeness for bananas and tomatoes.

The seven classes and their corresponding labels are:


banana_overripe (0) 


banana_ripe (1) 


banana_rotten (2) 


banana_unripe (3) 


tomato_fully_ripened (4) 


tomato_green (5) 


tomato_half_ripened (6) 

Class Imbalance and Augmentation
Initially, the dataset had a significant class imbalance, with banana classes having over 1500 images each, while tomato classes had fewer than 200 images. To address this, data augmentation techniques were used to increase the number of samples in the minority tomato classes. Augmentation methods included horizontal/vertical flips, rotations, brightness adjustments, and zooming. Additionally, class weighting was applied during training to help the model focus on the underrepresented classes.

Model Architectures
Three different neural network architectures were implemented and evaluated in this project:

1. Hybrid CNN–Transformer Model
This model combines a Convolutional Neural Network (CNN) for local feature extraction with a Vision Transformer (ViT) for global context modeling. The CNN extracts spatial features, which are then reshaped into patches for the transformer module. The transformer uses multi-head self-attention to understand global relationships between these patches. The model achieved the highest accuracy and robust generalization among the three architectures.


Optimizer: AdamW 


Loss Function: Categorical Cross-Entropy 


Metrics: Accuracy, Precision, Recall, F1-Score 

2. AlexNet Model
A modified AlexNet architecture was used, consisting of five convolutional layers and three fully connected layers to extract hierarchical features. The network uses ReLU activation, dropout regularization, and max pooling to reduce overfitting.


Input Size: 224
times224
times3 (RGB image) 


Optimizer: AdamW 


Loss Function: Categorical Cross-Entropy 


Metrics: Accuracy, Precision, Recall, F1-Score 

3. Inception Model
This architecture is a simplified Inception network known for its multi-path feature extraction. It uses a custom Inception module with parallel "towers" to process the input through multiple receptive fields, capturing both fine and coarse features. This model is compact and efficient, making it suitable for real-time systems.


Input Size: 224
times224
times3 (RGB image) 


Optimizer: AdamW 


Loss Function: Categorical Cross-Entropy 


Metrics: Accuracy, Precision, Recall, F1-Score 

Results and Comparison
The Hybrid CNN–Transformer model outperformed both the AlexNet and Inception models. The Hybrid model achieved a validation accuracy of 95% and a test accuracy of 95%. AlexNet had a validation accuracy of 91% and a test accuracy of 93% , while the Inception model had the lowest performance with a validation accuracy of 77% and a test accuracy of 78%.

Model	Training Accuracy	Validation Accuracy	Test Accuracy
CNN–Transformer	0.9806	0.95	0.95
AlexNet	0.9204	0.91	0.93
Inception	0.7457	0.77	0.78

Export to Sheets
The CNN–Transformer model demonstrated the highest accuracy and most stable generalization, while the AlexNet model showed some signs of overfitting. The Inception model provided a good balance of efficiency but did not achieve high accuracy.
