# Forest Cover Type Classification

Compared 16 classifiers on a 7-class imbalanced dataset and showed that accuracy and
ROC-AUC can look strong while recall on the rarest classes is zero.

The task is to predict the dominant tree species of 30×30 m land cells in the Roosevelt
National Forest (Colorado) from cartographic features: elevation, slope, distances to
water and roads, wilderness area and soil type. The data is the full
[Forest Cover Type](https://scikit-learn.org/stable/datasets/real_world.html#forest-covertypes)
dataset, 581,012 samples, with no subsampling. Each model is tuned on a separate
subsample, refit on the full training set and scored once on the same held-out test set.
The best of the 16, the Bagging Classifier, reaches a weighted F1 of 0.9693 (bootstrap
95% CI [0.9682, 0.9702]).

> University of Cagliari — coursework project, reworked.

**Full technical report — English: [PDF](report_en.pdf) · [Markdown](report_en.md) — Italian
edition: [PDF](report_it.pdf) · [Markdown](report_it.md)**

## Per-class results

![Per-class recall for all 16 models on the test set](figures/per_class_recall_heatmap.png)

The heatmap gives recall per class for all 16 models, in leaderboard order. AdaBoost
reaches 0.67 accuracy and 0.88 macro-AUC, and its recall on classes 4 to 7 is 0.00: 
it never predicts them. Neither aggregate metric shows this.

Every model is therefore also reported class by class: precision, recall and F1 for the
seven cover types, row-normalised confusion matrices, macro F1 next to weighted F1 in
each table, and one-vs-rest precision-recall curves for five representative models.

## Data

Seven cover types with a strong imbalance. The largest class holds 48.8% of the data and
the smallest 0.47% (Cottonwood/Willow), with Aspen at 1.63%. Weighted F1 is the reference
metric because it accounts for class size, and macro F1 is reported beside it because it
does not. The gap between the two is where failure on the rare classes appears.

## Method

- **Feature engineering:** The 44 one-hot columns (4 wilderness areas and 40 soil types)
  are folded back into 2 native categorical features, and 3 more are derived from
  elevation and vertical distance to hydrology. Final set: 15 features.
- **Two preprocessing paths:** Linear and distance-based models receive `StandardScaler`
  and `OneHotEncoder`. Tree-based models receive the raw features, unscaled.
- **Imbalance handling:** Class weights, searched as a hyperparameter for the tree and
  boosting models, and SMOTE for the MLP, both compared against training on the full data.
- **Evaluation:** Per-class report and row-normalised confusion matrix for every model,
  weighted and macro F1, one-vs-rest ROC/AUC and precision-recall curves, a 1,000-iteration
  bootstrap confidence interval on the winner, and permutation feature importance.

## Protocol

- All 16 models use the same stratified 80/20 split. The test set, 116,203 samples, is
  used once, for the final score.
- Hyperparameters are searched with `HalvingGridSearchCV`, or `HalvingRandomSearchCV` for
  LightGBM, on a stratified 100k subsample of the training set. The best estimator is then
  refit on the full 464,809-sample training set. The SVM stays on the 100k subsample
  because of its O(n²) cost.
- Nothing is selected or tuned on the test set and the winner is identified after all 16
  models have been scored.
- The full run writes every number it prints to `run_log_full.txt`, which is included in this repository.

## Results

Test set, 116,203 samples:

| Model | F1 (weighted) | F1 (macro) | Accuracy | AUC-macro |
|---|---|---|---|---|
| **Bagging Classifier** | **0.969** | **0.945** | 0.969 | 0.999 |
| Random Forest | 0.965 | 0.939 | 0.966 | 0.999 |
| KNN | 0.941 | 0.903 | 0.941 | 0.944 |
| Gradient Boosting | 0.919 | 0.904 | 0.919 | 0.991 |
| Decision Tree (pruned) | 0.908 | 0.862 | 0.909 | 0.958 |
| LightGBM | 0.877 | 0.855 | 0.875 | 0.989 |
| XGBoost | 0.862 | 0.844 | 0.863 | 0.985 |
| MLP (neural net) | 0.861 | 0.832 | 0.859 | 0.985 |
| … | | | | |
| Naive Bayes | 0.113 | 0.144 | 0.122 | 0.811 |

![Weighted F1 and macro F1 for all 16 models](figures/model_comparison.png)

The two panels rank the same models by weighted and by macro F1, and the two orders
differ. Gradient Boosting sits below KNN on weighted F1 and above it on macro F1, so a
model in the middle of the weighted ranking can be doing better on the rare classes than
the model above it.

The winning model, the Bagging Classifier, reaches weighted F1 0.9693, macro F1 0.9452
and a bootstrap 95% CI of [0.9682, 0.9702]. Its errors fall mainly where the classes
overlap:

![Row-normalised confusion matrix for the winning model](figures/cm_Bagging_Classifier.png)

Tree ensembles and distance-based models score above the linear and parametric ones,
because the geographic features form non-linear boundaries. And on the rare classes,
training on the full dataset helped more than the resampling strategies tried here.

## Limitations

- **Spatial autocorrelation:** Covertype cells are geographically contiguous, so a random
  split puts neighbouring cells on both sides of it. Absolute scores are optimistic and
  the clearest symptom is a nearest-neighbour model scoring 0.941. A spatially blocked
  split would be the stricter protocol. These numbers hold for the random-split benchmark
  the dataset is normally used with.
- **Tuning subset:** The hyperparameter search ran on a 100k subsample rather than the
  full training set, and the SVM was trained on 100k throughout. A larger subset would
  most likely move the mid-table models more than the leaders.
- **Single split:** The bootstrap interval measures test-set sampling noise, not
  split-to-split variance. There is no repeated or nested cross-validation on the full
  data.
- **Environment-dependent results:** Re-running the same code on a different CPU and
  library stack shifts the results in the third to fourth decimal, but the
  rankings and the conclusions are unaffected.

## Repository

```
forest_cover_type_classification.ipynb   # full pipeline: EDA → preprocessing → 16 models → evaluation
figures/                                 # every figure the notebook produces
report_en.md / report_en.pdf             # full technical report, English
report_it.md / report_it.pdf             # full technical report, Italian
run_log_full.txt                         # console log of the full run (581k samples)
run_log_25k.txt                          # console log of the 25k-subset run
requirements.txt
LICENSE
```

## Reproducing the run

The notebook downloads the dataset itself through `sklearn.datasets.fetch_covtype`, so
there is no data file to fetch by hand.

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook forest_cover_type_classification.ipynb
```

Set `USE_SUBSET = True` in the first code cell for a 25k-sample run.
`USE_SUBSET = False` reproduces the full run and needs more memory than a typical laptop has.
Python 3.11: the versions pinned in `requirements.txt` are the ones the published logs were produced with.

## Author

Samuele Nonnis — BSc student in Applied Computer Science and Data Analytics,
University of Cagliari.

[LinkedIn](https://linkedin.com/in/samuelenonnis)
