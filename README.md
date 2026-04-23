# 2026 Data Science HW2 - Bankruptcy Prediction Project

## Overview
This repository contains a comprehensive machine learning pipeline designed to predict corporate bankruptcy based on financial ratio data. The project addresses common real-world data challenges, including missing values, high-dimensional feature spaces, and extreme class imbalance (where bankrupt companies are the minority).

The core objective is to maximize the **Macro-F1 Score**, ensuring the model performs well on both healthy and bankrupt companies.

## Key Features
- **Advanced Imputation**: Utilizes `IterativeImputer` with `BayesianRidge` (MICE) to estimate missing financial data based on feature correlations.
- **Hybrid Resampling**: Implements a sophisticated pipeline of `BorderlineSMOTE` (to synthesize minority samples at the decision boundary) followed by `EditedNearestNeighbours` (to clean noisy overlaps).
- **Feature Engineering**: Combines `SelectKBest` (Mutual Information) with **PCA (Principal Component Analysis)** auxiliary features to capture both local indicators and global variance.
- **Hyperparameter Optimization**: Uses `Optuna` with Bayesian Optimization to tune XGBoost, LightGBM, and CatBoost.
- **Ensemble Learning**: Employs a `StackingClassifier` with a `LogisticRegression` meta-learner to blend the strengths of multiple Gradient Boosting Decision Trees (GBDT).

## Technical Pipeline

### 1. Data Preprocessing
- **Data Splitting**: Stratified splitting to preserve the minority class ratio in training and validation sets.
- **Missing Value Handling**: Statistical imputation using iterative techniques to maintain data distribution integrity.
- **Standardization**: Features are scaled using `StandardScaler` to ensure PCA and distance-based components perform correctly.

### 2. Feature Selection & Extraction
- **Mutual Information**: Selecting the Top-K features that share the most information with the target variable.
- **Auxiliary PCA**: Adding the first 5 principal components as extra features to help the linear meta-learner identify global patterns.

### 3. Handling Class Imbalance
- **Pipeline**:
    1. `BorderlineSMOTE`: Generates synthetic bankrupt samples.
    2. `ENN`: Removes ambiguous samples where the majority and minority classes overlap significantly.

### 4. Model Architecture
A three-model stacking ensemble is utilized:
- **XGBoost**: High-performance gradient boosting.
- **LightGBM**: Fast, leaf-wise growth boosting.
- **CatBoost**: Optimized for categorical stability and robust default performance.
- **Meta-Learner**: A Logistic Regression model that learns the optimal weights for the three base models.

### 5. Evaluation
The model is evaluated using the **Macro-F1 Score**. Probability thresholds are optimized post-training to find the best balance between Precision and Recall for the bankrupt class.

## Requirements
To run the notebook, ensure you have the following libraries installed:
- `numpy`, `pandas`
- `scikit-learn`
- `imbalanced-learn`
- `xgboost`, `lightgbm`, `catboost`
- `optuna`
- `matplotlib`, `seaborn`

## Usage
1. Place `train.csv` and `test.csv` in the root/data directory.
2. Run the `main.ipynb` notebook sequentially.
3. The final predictions `submission.csv` will be exported to root/result, formatted for Kaggle (columns: `Id`, `Bankrupt`).

## Project Structure
- `main.ipynb`: The primary execution script containing all logic from data loading to final CSV export.
- `submission.csv`: (Generated) The final prediction file.
- `README.md`: Project documentation.

---
**Author:** Joanna
**Course:** Data Science - Homework 2