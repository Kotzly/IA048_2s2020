import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score, accuracy_score, balanced_accuracy_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def load_features(filepath):
    with open(filepath) as file:
        content = file.read()
    lines = content.split("\n")
#     print(len(lines))
    for i in range(len(lines)):
        line = lines[i].split(" ")
        lines[i] = [float(x) for x in line if x]

    lines = [line for line in lines if line]    
    return np.array(lines)

def plot_confusion_matrix(cf_matrix):
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')

def plot_paired_histogram(x1, x2, l1="Sem normalização", l2="Normalização por Média e Desvio Padrão", n=5):
    fig, axes = plt.subplots(n, 2, figsize=(16, n*3))
    for i in range(n):
        axes[i, 0].hist(x1[:, i], bins=50)
        axes[i, 0].set_title(l1)

        axes[i, 1].hist(x2[:, i], bins=50)
        axes[i, 1].set_title(l2)

        axes[i, 0].set_ylabel(f"Feature {i}")
    plt.tight_layout()

def softmax(x):
    exp = np.exp(x)
    return exp/exp.sum(axis=1).reshape(-1, 1)

def macro_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

class OVA_LogisticRegression():
    
    def __init__(self, n_classes):
        pass
    
    def fit(self, x, y):
        self.classes = np.unique(y)
        self.estimators = []
        self.estimators_dict = {}
            
        for class_ in self.classes:
            print("Fitting Logistic Regression for class", class_)
            label = (y == class_)
            estimator = LogisticRegression(penalty="none", max_iter=10000)
            estimator.fit(x, label)
            self.estimators.append(estimator)
            self.estimators_dict[class_] = estimator
    
    def predict_proba(self, x):
        predictions = []
        for class_, estimator in self.estimators_dict.items():
            pred = estimator.predict_proba(x)[:, 1]
            predictions.append(pred)
        predictions = np.array(predictions).T
        return predictions

    def predict(self, x):
        pred = self.predict_proba(x)
        argmax = np.argmax(softmax(pred), axis=1)
        return self.classes[argmax]