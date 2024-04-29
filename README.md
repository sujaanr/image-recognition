# Handwritten Digit Recognition Using R

## Project Overview
This project focuses on the development of a handwritten digit recognition system using the MNIST dataset. The MNIST (Modified National Institute of Standards and Technology) database is a large collection of handwritten digits that is widely used for training and testing image processing systems in the field of machine learning.

## Dataset
The MNIST database contains 60,000 training images and 10,000 testing images. These images are derived from a mix of NIST's original datasets, which included American Census Bureau employees and American high school students. The images have been normalized and anti-aliased to fit into a 28x28 pixel bounding box, introducing grayscale levels.

### Extended MNIST (EMNIST)
The EMNIST dataset is the successor to the MNIST database, including not only digits but also handwritten uppercase and lowercase letters. This dataset adheres to the same 28x28 pixel format and preprocessing steps as MNIST, ensuring compatibility with tools designed for the original MNIST dataset.

## Methods
This project utilizes several supervised machine learning algorithms for digit recognition:

- **Decision Trees**
- **Random Forest**
- **Naive Bayes**
- **K-Nearest Neighbors (KNN)**
- **Gradient Boosting Machines (GBM)**
- **Support Vector Machines (SVM)**
- **Neural Networks**

Each model is trained on the training set and evaluated on the testing set to gauge its effectiveness at recognizing handwritten digits.

## Installation
To run the project, R and the following R packages are required:
- `caret`: For machine learning algorithms
- `rpart`: For decision trees
- `randomForest`: For random forest algorithm
- `e1071`: For naive Bayes classifier
- `kknn`: For k-nearest neighbors
- `gbm`: For gradient boosting
- `kernlab`: For support vector machines
- `nnet`: For neural networks

