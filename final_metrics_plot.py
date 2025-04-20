from metrics_plot import calculate_metrics, plot_metrics
import pickle
from sklearn.datasets import load_digits
import numpy as np

# Load test data
digits = load_digits()
X = digits.data
y = digits.target

# Split test data
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

metrics_dict = {}

# --- SVM ---
with open('MNIST_SVM.pickle', 'rb') as f:
    clf = pickle.load(f)
metrics_dict['SVM'] = calculate_metrics(clf, X_test, y_test)

# --- KNN ---
with open('MNIST_KNN.pickle', 'rb') as f:
    clf = pickle.load(f)
metrics_dict['KNN'] = calculate_metrics(clf, X_test, y_test)

# --- RFC ---
with open('MNIST_RFC.pickle', 'rb') as f:
    clf = pickle.load(f)
metrics_dict['RFC'] = calculate_metrics(clf, X_test, y_test)

# Print and Plot
for model, metrics in metrics_dict.items():
    print(f"\n{model} Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

plot_metrics(metrics_dict)
