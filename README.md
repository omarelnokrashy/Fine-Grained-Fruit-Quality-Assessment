# Fine-Grained Fruit Quality Assessment

## Project Overview

[cite_start]This project is a deep learning-based image classification system designed to perform fine-grained classification of fruit quality[cite: 7]. [cite_start]The model analyzes image features to assess and classify the ripeness levels and spoilage signs of bananas and tomatoes[cite: 8]. [cite_start]The classification system includes seven distinct categories: banana overripe, banana ripe, banana rotten, banana unripe, tomato fully ripened, tomato green, and tomato half-ripened[cite: 9].

[cite_start]This project was completed as part of a Neural Networks course[cite: 3].

## Team

The project team members and their IDs are:
* [cite_start]Omar Elsayed Ibrahim Aly (ID: 2022170827) [cite: 5]
* [cite_start]Omar Mohamed Adel Salama (ID: 2022170829) [cite: 5]
* [cite_start]Aly Tarek Fekry (ID: 2022170825) [cite: 5]
* [cite_start]Malak Amr Ismail (ID: 2022170843) [cite: 5]
* [cite_start]Malak Hossam Aboelfetouh (ID: 2022170842) [cite: 5]
* [cite_start]Maria Raafat Ezra (ID: 2022170835) [cite: 5]

## Dataset

[cite_start]The dataset consists of labeled food images representing different stages of ripeness and quality for bananas and tomatoes[cite: 11]. [cite_start]Each image is assigned to one of seven classes[cite: 12, 13].

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

[cite_start]The initial dataset had a significant class imbalance[cite: 15]. [cite_start]To address this, targeted data augmentation was applied to increase the number of samples in the minority classes[cite: 19]. [cite_start]Augmentation methods included horizontal/vertical flips, rotations, brightness adjustments, and zooming[cite: 24]. [cite_start]In addition, class weighting was used during model training to penalize errors from minority classes more heavily[cite: 29]. [cite_start]A custom function was implemented to calculate class weights dynamically from the label distribution using TensorFlow[cite: 30].

## Model Architectures

[cite_start]Three different deep learning models were developed and compared for this project[cite: 35].

### 1. Hybrid CNN–Transformer Model

[cite_start]This hybrid model combines a Convolutional Neural Network (CNN) for feature extraction with a Vision Transformer (ViT) for classification[cite: 37]. [cite_start]The CNN captures local spatial features, and these features are then reshaped into a sequence of patches for the transformer module[cite: 38, 39]. [cite_start]The Vision Transformer uses multi-head self-attention to capture global relationships[cite: 40]. [cite_start]The final classification is done through a fully connected layer[cite: 42].

* [cite_start]**Optimizer**: AdamW [cite: 44]
* [cite_start]**Loss Function**: Categorical Cross-Entropy [cite: 45]
* [cite_start]**Regularization**: Dropout, Early Stopping, Learning Rate Scheduling [cite: 46]
* [cite_start]**Metrics**: Accuracy, Precision, Recall, F1-Score [cite: 47]

### 2. AlexNet Model

[cite_start]The AlexNet model was adapted for this classification task[cite: 54]. [cite_start]The architecture uses a series of convolutional and max pooling layers, which progressively extract hierarchical features from the input image[cite: 55]. [cite_start]The model incorporates ReLU activation functions, dropout regularization, and max pooling to reduce overfitting and manage computational complexity[cite: 57]. [cite_start]The final output layer uses a softmax activation to produce class probabilities[cite: 58].

* [cite_start]**Input Size**: $224 \times 224 \times 3$ (RGB image) [cite: 60]
* [cite_start]**Optimizer**: AdamW [cite: 73]
* [cite_start]**Loss Function**: Categorical Cross-Entropy [cite: 74]
* [cite_start]**Regularization**: Dropout, Early Stopping, Learning Rate Scheduling [cite: 75]
* [cite_start]**Metrics**: Accuracy, Precision, Recall, F1-Score [cite: 76]

### 3. Inception Model

[cite_start]This architecture is a simplified version of the Inception network, which is known for its multi-path feature extraction[cite: 82, 83]. [cite_start]A custom Inception module with three parallel "towers" was designed to process the input through multiple receptive fields and pooling paths[cite: 85, 89, 90, 91]. [cite_start]The model is compact and efficient, achieving competitive performance while maintaining a low parameter count[cite: 101].

* [cite_start]**Input Size**: $224 \times 224 \times 3$ (RGB image) [cite: 87]
* [cite_start]**Optimizer**: AdamW [cite: 97]
* [cite_start]**Loss Function**: Categorical Cross-Entropy [cite: 98]
* [cite_start]**Regularization**: Dropout, Early Stopping, Learning Rate Scheduling [cite: 99]
* [cite_start]**Metrics**: Accuracy, Precision, Recall, F1-Score [cite: 100]

## Results and Comparison

[cite_start]All three models were trained and evaluated under similar conditions to ensure a fair comparison[cite: 107, 108]. [cite_start]The Hybrid CNN–Transformer model achieved the highest accuracy and showed robust generalization, likely due to its ability to capture both local and global features[cite: 112].

| Model | Training Accuracy | Validation Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- |
| CNN–Transformer | 0.9806 | 0.95 | 0.95 |
| AlexNet | 0.9204 | 0.91 | 0.93 |
| Inception | 0.7457 | 0.77 | 0.78 |

[cite_start]The AlexNet model performed well in early training but showed signs of overfitting with deeper layers[cite: 113]. [cite_start]The Inception model offered a good balance of accuracy and computational efficiency, but did not achieve high accuracy[cite: 114].
