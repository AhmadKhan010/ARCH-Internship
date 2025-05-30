# Regression and Classification Deep Dive

This project implements Task 02 of the ARCH Technologies Machine Learning Internship (Category B), focusing on regression and classification models as outlined in Chapter 4 of Hands-On Machine Learning. It includes model training, evaluation, and optimization for regression (synthetic quadratic data) and classification (Iris dataset), along with a Gradio web app for interactive predictions.

## 📁 Structure

- `notebooks`: includes all the python notebooks which have code
- `dataset/`: includes the dataset being used
- `Reports/`: Includes report of the project

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/AhmadKhan010/mnist-digit-recognizer.git
cd mnist-digit-recognizer
```

2. **Create and Activate Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Train the Models**

- Open the notebook and run it:

```bash
cd notebooks
jupyter notebook Custom-Dataset-Implementation.ipynb
```

# 📝 Project Details

### Datasets:

- Regression: Synthetic quadratic data (100 samples, y = 0.5x² + x + 2 + noise).
- Classification: Iris dataset (150 samples, 3 classes, using petal length/width).

### Models:

- Regression: Linear Regression, Ridge, Lasso, Elastic Net (degree=2 polynomial features).
- Classification: Logistic Regression (OvR), Softmax Regression.

### Metrics:

- Regression: RMSE, R².
- Classification: Accuracy, F1-score.
- Additional: Training time, feature importance (coefficients).
  
### Visualizations:

- Learning curves for Linear and Polynomial Regression.
- Coefficient analysis across models.
-- Validation loss for custom Softmax Regression (Exercise 12).

Custom Implementation: Batch Gradient Descent with early stopping for Softmax Regression (Chapter 4, Exercise 12).
