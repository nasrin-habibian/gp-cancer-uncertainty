# Gaussian Process classification with uncertainty on breast cancer diagnostic data

A small methods project: can a Gaussian Process (GP) classifier give useful
*uncertainty* about a diagnosis-style prediction, and can that uncertainty be used
to decide when a model should abstain and defer to a human expert?

I come from numerical analysis and optimisation, and I have used Gaussian Process
surrogates in earlier projects (see my
[warm-start Newton-Broyden](https://github.com/nasrin-habibian/warmstart-newton-broyden) and
[building-energy](https://github.com/nasrin-habibian/building-energy-ai-optimisation) repositories).
This project applies the same tool to a biomedical benchmark as a first step into
that field.

> **Scope note.** This is a methods demonstration on a small public benchmark
> dataset. It is not a clinical tool and makes no clinical claims.

## Data

Breast Cancer Wisconsin (Diagnostic), bundled with scikit-learn
(`sklearn.datasets.load_breast_cancer`), so nothing needs to be downloaded.
569 samples (212 malignant, 357 benign), 30 numeric features computed from
digitised images of fine-needle aspirates of breast masses.

## Method

- Models: logistic regression (baseline), random forest, and a GP classifier
  with a constant x RBF kernel (kernel hyperparameters fitted by maximising the
  Laplace-approximated marginal likelihood, with 2 optimiser restarts).
- Features standardised inside each cross-validation fold (via a pipeline) to avoid leakage.
- Evaluation: stratified 5-fold cross-validation, using out-of-fold predicted
  probabilities. Metrics: accuracy, ROC AUC, Brier score, log loss.
- Uncertainty analysis:
  1. Calibration curves.
  2. Selective prediction: rank cases by confidence (distance of the predicted
     probability from 0.5), let the model abstain on the least confident ones, and
     measure accuracy on the cases it answers.
  3. A 70/30 hold-out check of where the GP's errors fall.

## Results

Out-of-fold, 5-fold CV (seed 42):

| Model | Accuracy | ROC AUC | Brier | Log loss |
|---|---|---|---|---|
| Logistic regression | 0.9736 | 0.9947 | 0.0200 | 0.0764 |
| Random forest | 0.9525 | 0.9882 | 0.0336 | 0.1776 |
| Gaussian process | 0.9684 | 0.9952 | 0.0241 | 0.0910 |

Main observations:

- **The GP is not better than logistic regression here.** Accuracy and log loss
  are slightly worse, AUC is marginally higher. On this dataset the classes are
  close to linearly separable, so a simple linear model is already very strong.
  I would not claim a GP advantage from these numbers.
- **Abstaining on uncertain cases helps.** Answering only the 80% most confident
  cases raises accuracy from 0.968 to 0.998 for the GP (and from 0.974 to 0.998 for
  logistic regression). Uncertainty ranking is doing useful work, but this is
  equally true of the logistic baseline.
- **Errors concentrate in uncertain cases.** In the 70/30 hold-out check
  (171 test cases), 22 cases had predicted probability between 0.2 and 0.8. Of
  the GP's 5 errors, 4 were in that uncertain group and 1 was among the confident cases.
  This is a small sample, so it is suggestive rather than conclusive.
- **Calibration** is reasonable for all three models, with mild over- and
  under-confidence in the middle range (`results/calibration.png`). With this
  little data the curves are noisy.

## Limitations

- One small, relatively easy benchmark. Results may not transfer to harder or
  noisier biomedical data.
- The GP uses one isotropic kernel across all 30 features. An ARD kernel or
  feature selection could change the picture, but is more expensive to fit.
- No external validation, and no hyperparameter search for the baselines.
- The 0.2-0.8 "uncertain" band on the hold-out split is a convenient choice, not a tuned threshold.

## Possible next steps

- Try an ARD kernel or sparse GP, and compare against a properly tuned
  gradient-boosting model.
- Test on a harder dataset, for example radiomics or gene-expression data from
  a public cancer repository, where the linear baseline is less dominant.
- Evaluate uncertainty more formally (expected calibration error, conformal prediction).

## Reproduce

```bash
pip install -r requirements.txt
python gp_uncertainty.py
```

Outputs are written to `results/` (`metrics.csv`, `gp_holdout_summary.csv`,
`roc_curves.png`, `calibration.png`, `selective_prediction.png`).
