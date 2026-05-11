import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import shap
import warnings

def evaluate_model_cv(X: pd.DataFrame, y: pd.Series, n_splits=100) -> dict:
    """Évalue la régression logistique via validation croisée et retourne les scores AUC."""
    warnings.filterwarnings("ignore")
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    clf = LogisticRegression(random_state=42, max_iter=500)
    scaler = StandardScaler()

    classes = np.unique(y)
    auc_scores = {c: [] for c in classes}

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        clf.fit(X_train_scaled, y_train)
        probas_val = clf.predict_proba(X_val_scaled)

        for i, class_label in enumerate(clf.classes_):
            y_val_bin = (y_val == class_label).astype(int)
            auc = roc_auc_score(y_val_bin, probas_val[:, i])
            auc_scores[class_label].append(auc)

    return auc_scores

def plot_shap_summary(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series):
    """Entraîne un modèle final et génère le graphique SHAP global."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    clf = LogisticRegression(random_state=42, max_iter=500)
    clf.fit(X_train_scaled, y_train)

    explainer = shap.LinearExplainer(clf, X_train_scaled)
    shap_values = explainer(X_test_scaled)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test_scaled,
        plot_type="bar",
        class_names=clf.classes_,
        show=False
    )
    plt.title("Importance globale des variables par classe (SHAP)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()
