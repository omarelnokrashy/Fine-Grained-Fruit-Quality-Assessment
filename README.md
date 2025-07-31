# Fine-Grained Fruit Quality Assessment

## Project Overview

This project is a deep learning-based image classification system designed to perform fine-grained classification of fruit quality. The model analyzes image features to assess and classify the ripeness levels and spoilage signs of bananas and tomatoes. The classification system includes seven distinct categories: banana overripe, banana ripe, banana rotten, banana unripe, tomato fully ripened, tomato green, and tomato half-ripened.

This project was completed as part of a Neural Networks course.

## Team

The project team members and their IDs are:
* Omar Elsayed Ibrahim Aly (ID: 2022170827)
* Omar Mohamed Adel Salama (ID: 2022170829)
* Aly Tarek Fekry (ID: 2022170825)
* Malak Amr Ismail (ID: 2022170843)
* Malak Hossam Aboelfetouh (ID: 2022170842)
* Maria Raafat Ezra (ID: 2022170835)

## Dataset

The dataset consists of labeled food images representing different stages of ripeness and quality for bananas and tomatoes. Each image is assigned to one of seven classes.

The class names and their corresponding labels are:

| Class Name | Label |
| :--- | :--- |
| banana_overripe | 0 |
| banana_ripe | 1 |
| banana_rotten | 2 |
| banana_unripe | 3 |
| tomato_fully_ripened | 4 |
| tomato_green | 5 |
| tomato_half-ripened | 6 |

### Addressing Class Imbalance

The initial dataset had a significant class imbalance. To address this, targeted data augmentation was applied to increase the number of samples in the minority classes. Augmentation methods included horizontal/vertical flips, rotations, brightness adjustments, and zooming. In addition, class weighting was used during model training to penalize errors from minority classes more heavily. A custom function was implemented to calculate class weights dynamically from the label distribution using TensorFlow.

## Model Architectures

Three different deep learning models were developed and compared for this project.

### 1. Hybrid CNN–Transformer Model

This hybrid model combines a Convolutional Neural Network (CNN) for feature extraction with a Vision Transformer (ViT) for classification. The CNN captures local spatial features, and these features are then reshaped into a sequence of patches for the transformer module. The Vision Transformer uses multi-head self-attention to capture global relationships. The final classification is done through a fully connected layer.

* **Optimizer**: AdamW
* **Loss Function**: Categorical Cross-Entropy
* **Regularization**: Dropout, Early Stopping, Learning Rate Scheduling
* **Metrics**: Accuracy, Precision, Recall, F1-Score

### 2. AlexNet Model

The AlexNet model was adapted for this classification task. The architecture uses a series of convolutional and max pooling layers, which progressively extract hierarchical features from the input image. The model incorporates ReLU activation functions, dropout regularization, and max pooling to prevent overfitting. The final output layer uses a softmax activation to produce class probabilities.

* **Input Size**: $224 \times 224 \times 3$ (RGB image)
* **Optimizer**: AdamW
* **Loss Function**: Categorical Cross-Entropy
* **Regularization**: Dropout, Early Stopping, Learning Rate Scheduling
* **Metrics**: Accuracy, Precision, Recall, F1-Score

### 3. Inception Model

This architecture is a simplified version of the Inception network, which is known for its multi-path feature extraction. A custom Inception module with three parallel "towers" was designed to process the input through multiple receptive fields and pooling paths. The model is compact and efficient, achieving competitive performance while maintaining a low parameter count.

* **Input Size**: $224 \times 224 \times 3$ (RGB image)
* **Optimizer**: AdamW
* **Loss Function**: Categorical Cross-Entropy
* **Regularization**: Dropout, Early Stopping, Learning Rate Scheduling
* **Metrics**: Accuracy, Precision, Recall, F1-Score

## Results and Comparison

All three models were trained and evaluated under similar conditions to ensure a fair comparison. The Hybrid CNN–Transformer model achieved the highest accuracy and showed robust generalization, likely due to its ability to capture both local and global features.

| Model | Training Accuracy | Validation Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- |
| CNN–Transformer | 0.9806 | 0.95 | 0.95 |
| AlexNet | 0.9204 | 0.91 | 0.93 |
| Inception | 0.7457 | 0.77 | 0.78 |

The AlexNet model performed well in early training but showed signs of overfitting with deeper layers. The Inception model offered a good balance of accuracy and computational efficiency, but did not achieve high accuracy.
