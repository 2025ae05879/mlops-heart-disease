# Model comparison

CV = StratifiedKFold(5, shuffle, seed=523) on the training split (n=242); test = held-out stratified 20% (n=61), threshold 0.5.

| model         | cv_accuracy     | cv_precision    | cv_recall       | cv_f1           | cv_roc_auc      |   test_accuracy |   test_precision |   test_recall |   test_f1 |   test_roc_auc |   fit_s | best_params                                                                          |
|:--------------|:----------------|:----------------|:----------------|:----------------|:----------------|----------------:|-----------------:|--------------:|----------:|---------------:|--------:|:-------------------------------------------------------------------------------------|
| logreg        | 0.859 +/- 0.034 | 0.870 +/- 0.055 | 0.820 +/- 0.048 | 0.843 +/- 0.036 | 0.910 +/- 0.025 |           0.836 |            0.821 |         0.821 |     0.821 |          0.913 |     5.1 | {"C": 0.3, "class_weight": null, "l1_ratio": 0.0}                                    |
| random_forest | 0.814 +/- 0.057 | 0.852 +/- 0.092 | 0.730 +/- 0.044 | 0.785 +/- 0.058 | 0.889 +/- 0.045 |           0.852 |            0.828 |         0.857 |     0.842 |          0.917 |     8.1 | {"max_depth": 3, "max_features": "sqrt", "min_samples_leaf": 5, "n_estimators": 500} |
| hist_gb       | 0.777 +/- 0.025 | 0.775 +/- 0.057 | 0.739 +/- 0.050 | 0.753 +/- 0.013 | 0.885 +/- 0.025 |           0.82  |            0.793 |         0.821 |     0.807 |          0.921 |     0.8 | {"l2_regularization": 1.0, "learning_rate": 0.03, "max_depth": 2, "max_iter": 300}   |

## Selected model

Shortlist (CV ROC-AUC within 0.02 of best 0.910): logreg. Chose 'logreg' for the highest CV recall (0.820) - false negatives are the costly error for a screening tool.
