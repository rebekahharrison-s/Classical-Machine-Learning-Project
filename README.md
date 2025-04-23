# Classical Machine Learning - Classification of Digits Dataset

This repository demonstrates the application of classical machine learning models for the classification of handwritten digits using the Digits dataset. The dataset consists of 8x8 pixel images representing digits (0-9) and has been processed using several well-known classifiers, including Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Decision Trees (DT), Random Forest (RF), Logistic Regression (LR), and Naïve Bayes (NB). The study explores the effects of dimensionality reduction using Principal Component Analysis (PCA) and hyperparameter tuning to optimise model performance.

## Contents

1. **Data Preprocessing**
   - The Digits dataset is loaded, and images are flattened into feature vectors. The data is split into training and testing sets (80%/20%).
   - Preprocessing steps include feature scaling using StandardScaler, and PCA is applied where necessary to assess its impact on performance.

2. **Machine Learning Models**
   - Various classifiers are implemented, trained, and evaluated. Models are evaluated both with and without PCA to compare the effects of dimensionality reduction.
     - **Support Vector Machine (SVM)**: Performs well in high-dimensional spaces and is capable of capturing complex decision boundaries.
     - **K-Nearest Neighbors (KNN)**: Suitable for small datasets and sensitive to local data structure.
     - **Decision Trees (DT)**: Interpretable and fast to train, but prone to overfitting on small datasets.
     - **Random Forest (RF)**: An ensemble of decision trees, robust against overfitting and effective for noisy data.
     - **Logistic Regression (LR)**: A simple model effective for binary classification and reasonable for multi-class tasks.
     - **Naïve Bayes (NB)**: A probabilistic classifier assuming feature independence, though it may struggle with correlated features.

3. **Dimensionality Reduction**
   - PCA is applied to reduce the number of features and improve computational efficiency, though it may remove important features if not carefully managed.
   - The impact of PCA on model performance is explored for each classifier.

4. **Hyperparameter Tuning**
   - Hyperparameter tuning using GridSearchCV is applied to optimise each model’s performance.
     - For SVM, KNN, Random Forest, and Logistic Regression, the grid search optimises key parameters like kernel type, number of neighbours, and tree depth.

5. **Model Evaluation**
   - Each model's performance is evaluated using:
     - **Accuracy**
     - **Balanced Accuracy**
     - **ROC-AUC**
     - **Confusion Matrices**
   - The models are compared to assess the effect of dimensionality reduction and hyperparameter tuning on overall performance.

6. **Visualisations**
   - Confusion matrices and ROC curves are plotted to visualise model performance before and after hyperparameter tuning.
   - **Confusion Matrices**: Show misclassifications and insights into model weaknesses.
   - **ROC Curves**: Provide a visual representation of model performance across various classification thresholds.

## Installation

To run this analysis, ensure you have the following libraries installed:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

## Usage

1. **Load Data**: Use the provided Python scripts to load the Digits dataset and split it into training and testing sets.
2. **Train Models**: Train various classifiers, both with and without PCA, and evaluate their performance.
3. **Hyperparameter Tuning**: Use GridSearchCV to optimise the hyperparameters for each model.
4. **Visualise Results**: Generate confusion matrices and ROC curves to evaluate model performance.
5. **Save Models**: Save the trained models using `joblib` for future use.

## Conclusion

The study demonstrates the effectiveness of various classical machine learning models for digit classification. Hyperparameter tuning and PCA both play significant roles in optimising model performance. Random Forest and SVM models achieved the highest accuracy, while Naïve Bayes and Decision Trees, though improving with tuning, performed comparatively worse.
