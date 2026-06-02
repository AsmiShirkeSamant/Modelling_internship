# Iris Species Classification Project

A machine learning project benchmarking multiple classifiers on the classic **Iris Flower Dataset**. The goal is to accurately predict the species of an iris flower (*Setosa, Versicolor, Virginica*) based on its sepal and petal measurements.

---

## Project Status
The pipeline has evolved from a simple notebook baseline into a multi-model benchmarking script.

---

## Completed Workflow

### 1. Data Acquisition & Exploratory Analysis (EDA)
- Ingested the dataset from Scikit-Learn into a structured `pandas` DataFrame.
- Used Seaborn's `pairplot` and overlaid histograms to map feature distributions across species.
- Discovered that *Setosa* is entirely linearly separable based on petal traits, while *Versicolor* and *Virginica* form a tight, overlapping boundary.

### 2. Data Engineering & Preprocessing
- Implemented strict data separation using a 75% training split and a 25% hold-out test split.
- Encapsulated `StandardScaler` inside Scikit-Learn `Pipeline` objects to ensure scaling parameters are computed solely within training folds, preventing data leakage.

### 3. Multi-Model Comparison
Implemented and evaluated 5 machine learning models:

| Model | Key Parameters |
|---|---|
| Logistic Regression | `C=1`, `max_iter=300` |
| K-Nearest Neighbors | `n_neighbors=5` |
| Decision Tree | `max_depth=4` |
| Random Forest | `n_estimators=100` |
| Naive Bayes | Gaussian density, full feature independence assumed |

### 4. Model Evaluation
- Ran 5-Fold Stratified Cross-Validation (`cross_val_score`) across all models to produce stable, split-agnostic accuracy estimates.
- Confirmed that all models tie at **92.11% test accuracy** (3 misclassifications out of 38 test samples), with all errors confined to the natural overlap region between *Versicolor* and *Virginica*.
- Cross-validation scores show more differentiation between models than the test set alone, given the small hold-out size.

---

## Tech Stack
- **Language:** Python
- **Libraries:** `scikit-learn`, `pandas`, `numpy`, `seaborn`, `matplotlib`

---

## Visualizations

| File | Description |
|---|---|
| `output_1.png` | Feature distribution histograms by species |
| `output_2.png` | Feature comparison pairplot |
| `output_3.png` | Model accuracy comparison bar chart |
| `output_4.png` | Confusion matrices for all models |
| `output_5.png` | Per-class F1 score heatmap |
| `output_6.png` | Random Forest feature importance |
| `output_7.png` | Cross-validation accuracy comparison |
