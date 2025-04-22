from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

def plot_metrics(metrics_dict):
    labels = list(next(iter(metrics_dict.values())).keys())
    models = list(metrics_dict.keys())
    data = [list(metrics_dict[model].values()) for model in models]

    x = range(len(labels))
    bar_width = 0.25

    plt.figure(figsize=(10, 6))
    for i in range(len(models)):
        plt.bar([p + bar_width*i for p in x], data[i], width=bar_width, label=models[i])

    plt.xticks([p + bar_width for p in x], labels)
    plt.ylabel('Score')
    plt.title('Model Comparison - Metrics')
    plt.legend()
    plt.tight_layout()
    plt.show()
