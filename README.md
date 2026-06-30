# Decodelab project 2
# Project 2: Iris Flower Classification using K-Nearest Neighbors (KNN)

## Overview
This project implements a supervised machine learning model to classify Iris flowers into one of three species — *Setosa*, *Versicolor*, and *Virginica* — based on four physical measurements: sepal length, sepal width, petal length, and petal width.

The model is built using the **K-Nearest Neighbors (KNN)** algorithm, a simple yet effective classification technique that predicts a sample's class based on the majority class among its closest neighbors in the feature space.

## Dataset
The project uses the **Iris dataset**, a classic benchmark dataset in machine learning, loaded directly via `sklearn.datasets.load_iris()`. It contains 150 samples (50 per species) with 4 numerical features each.

## Workflow
1. **Data Loading** – Imported the Iris dataset and separated features (`X`) from target labels (`y`).
2. **Train-Test Split** – Split the data into training (80%) and testing (20%) sets using `train_test_split`, with a fixed `random_state` for reproducibility.
3. **Feature Scaling** – Applied `StandardScaler` to normalize feature values, since KNN is a distance-based algorithm sensitive to feature magnitude.
4. **Model Training** – Trained a `KNeighborsClassifier` with `n_neighbors=5` on the scaled training data.
5. **Prediction & Evaluation** – Generated predictions on the test set and evaluated performance using:
   - Confusion Matrix
   - Classification Report (precision, recall, F1-score per class)
   - Weighted F1 Score

## Tools & Libraries
- Python 3
- scikit-learn (`sklearn`)
- PyCharm (IDE)

## Key Concepts Demonstrated
- Supervised classification
- Train-test splitting
- Feature scaling and its importance for distance-based algorithms
- Model evaluation using multiple metrics beyond simple accuracy

## How to Run
```bash
pip install scikit-learn
python "project 2.py"
```

## Results
The model achieves high classification accuracy on the test set, with strong precision and recall across all three Iris species, as reflected in the classification report and F1 score output.

---
*Part of the DecodeLabs Internship machine learning track.*
