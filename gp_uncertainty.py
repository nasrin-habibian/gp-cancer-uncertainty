"""
Gaussian Process classification with uncertainty on the Breast Cancer Wisconsin
(Diagnostic) dataset.

Question: can a Gaussian Process classifier give useful *uncertainty* about a
diagnosis-style prediction, and can that uncertainty be used to decide when the
model should abstain and defer to a human?

The dataset ships with scikit-learn, so no download is needed. It contains 569
samples with 30 numeric features computed from digitised images of fine-needle
aspirates of breast masses (label: malignant / benign).

This is a methods demonstration on a small benchmark dataset. It is NOT a
clinical tool and makes no clinical claims.

Run:  python gp_uncertainty.py
Output: results/ (metrics table + figures)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path("results")
OUT.mkdir(exist_ok=True)


def load_data():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target  # target: 1 = benign, 0 = malignant
    # Make "1" the clinically important class (malignant) for readability.
    y = 1 - y
    return X, y, data.target_names


def build_models():
    gp_kernel = C(1.0) * RBF(length_scale=5.0)
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300, random_state=SEED
        ),
        "Gaussian process": make_pipeline(
            StandardScaler(),
            GaussianProcessClassifier(
                kernel=gp_kernel, random_state=SEED, n_restarts_optimizer=2
            ),
        ),
    }


def evaluate(models, X, y):
    """Out-of-fold predicted probabilities via stratified 5-fold CV."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probs, rows = {}, []
    for name, model in models.items():
        p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        probs[name] = p
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y, p > 0.5),
                "roc_auc": roc_auc_score(y, p),
                "brier": brier_score_loss(y, p),
                "log_loss": log_loss(y, p),
            }
        )
    table = pd.DataFrame(rows).set_index("model").round(4)
    return probs, table


def selective_prediction(y, p, coverages=np.linspace(0.5, 1.0, 26)):
    """Accuracy when the model abstains on its least confident cases.

    Confidence = distance of the predicted probability from 0.5.
    coverage = fraction of cases the model chooses to answer.
    """
    confidence = np.abs(p - 0.5)
    order = np.argsort(-confidence)  # most confident first
    correct = ((p > 0.5).astype(int) == y.values)[order]
    accs = []
    for c in coverages:
        k = int(round(c * len(y)))
        accs.append(correct[:k].mean())
    return coverages, np.array(accs)


def plot_roc(y, probs):
    plt.figure(figsize=(5, 4.5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves (5-fold out-of-fold)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "roc_curves.png", dpi=200)
    plt.close()


def plot_calibration(y, probs):
    plt.figure(figsize=(5, 4.5))
    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    plt.xlabel("Mean predicted probability of malignant")
    plt.ylabel("Observed fraction malignant")
    plt.title("Calibration (5-fold out-of-fold)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "calibration.png", dpi=200)
    plt.close()


def plot_selective(y, probs):
    plt.figure(figsize=(5, 4.5))
    for name, p in probs.items():
        cov, acc = selective_prediction(y, p)
        plt.plot(cov, acc, marker=".", label=name)
    plt.xlabel("Coverage (fraction of cases the model answers)")
    plt.ylabel("Accuracy on answered cases")
    plt.title("Abstaining on the least confident cases")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "selective_prediction.png", dpi=200)
    plt.close()


def gp_on_holdout(X, y):
    """Fit the GP on a train split and inspect its uncertain cases on the test split."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    gp = build_models()["Gaussian process"].fit(X_tr, y_tr)
    p = gp.predict_proba(X_te)[:, 1]
    uncertain = (p > 0.2) & (p < 0.8)
    wrong = (p > 0.5).astype(int) != y_te.values

    summary = {
        "n_test": int(len(y_te)),
        "n_uncertain (0.2 < p < 0.8)": int(uncertain.sum()),
        "errors_total": int(wrong.sum()),
        "errors_among_uncertain": int((wrong & uncertain).sum()),
        "errors_among_confident": int((wrong & ~uncertain).sum()),
    }
    fitted_kernel = gp.named_steps["gaussianprocessclassifier"].kernel_
    return summary, str(fitted_kernel)


def main():
    X, y, target_names = load_data()
    print(f"Samples: {len(y)}, features: {X.shape[1]}, malignant: {int(y.sum())}, "
          f"benign: {int((1 - y).sum())}\n")

    models = build_models()
    probs, table = evaluate(models, X, y)
    table.to_csv(OUT / "metrics.csv")
    print("Cross-validated metrics (out-of-fold):")
    print(table.to_string(), "\n")

    plot_roc(y, probs)
    plot_calibration(y, probs)
    plot_selective(y, probs)

    # How much does abstaining help each model?
    print("Accuracy when answering only the 80% most confident cases:")
    for name, p in probs.items():
        cov, acc = selective_prediction(y, p, coverages=np.array([0.8, 1.0]))
        print(f"  {name:20s} full: {acc[1]:.4f}   top 80%: {acc[0]:.4f}")
    print()

    summary, kernel = gp_on_holdout(X, y)
    print("GP hold-out check (70/30 split):")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  learned kernel: {kernel}")
    pd.Series(summary).to_csv(OUT / "gp_holdout_summary.csv", header=["value"])
    print(f"\nFigures and tables saved to ./{OUT}/")


if __name__ == "__main__":
    main()
