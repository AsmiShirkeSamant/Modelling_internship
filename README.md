# Iris Species Classification Project

A production-grade machine learning project focused on exploring, benchmarking, and deploying a multi-class prediction model using the classic **Iris Flower Dataset**. The goal is to accurately predict the species of an iris flower (*Setosa, Versicolor, Virginica*) based on its sepal and petal measurements.

---

## Project Status:
The pipeline has evolved from a simple notebook baseline into a comprehensive multi-model benchmarking script.

---

## Completed Workflow:

1. **Data Acquisition & Exploratory Analysis (EDA)**
   * Ingested the dataset from Scikit-Learn into a structured `pandas` DataFrame.
   * Utilized Seaborn's `pairplot` and native pandas `.hist()` modules to map data distributions.
   * Discovered that *Setosa* is entirely linearly separable based on petal traits, while *Versicolor* and *Virginica* form a tight, overlapping boundary.

2. **Data Engineering & Preprocessing**
   * Implemented strict data separation using a 75% training split and a 25% hold-out test split.
   * Encapsulated `StandardScaler()` directly inside atomic Scikit-Learn pipelines via `make_pipeline`. 
   * This design protects the workflow from data leakage by calculating scaling parameters ($\mu, \sigma$) purely inside individual training folds.

3. **Multi-Model comparision**
   Implemented and evaluated 7 machine learning models with precise hyperparameter constraints:
   * Logistic Regression: Probabilistic classifier with multinomial Softmax optimization (`C=1`, `max_iter=300`).
   * K-Nearest Neighbors (KNN): Distance-based clustering balancing density flags (`n_neighbors=5`).
   * Support Vector Machine (SVM): High-margin hyperplane separator using a Radial Basis Function kernel (`kernel='rbf'`, `C=1`).
   * Decision Tree: Single rule-based classifier with locked depth control (`max_depth=4`) to stop overfitting.
   * Random Forest: Ensemble bagging framework using 100 bootstrapped trees (`n_estimators=100`).
   * Gradient Boosting: Sequential boosting trees optimized against multinomial cross-entropy residuals (`n_estimators=100`).
   * Naive Bayes: Gaussian density model assuming complete feature independence (`GaussianNB`).

4. **Model Evaluation**
   * Configured an out-of-fold 5-Fold Cross-Validation routine (`cross_val_score`) to generate highly stable model metrics and eliminate single-split lucky outliers.
   * Rendered a clean Matplotlib Accuracy Leaderboard Bar Chart showing average cross-validation accuracy paired with standard deviation error caps to grade model stability.
   * Ran a complete diagnostic error analysis using a custom multi-axis subplot function (`plot_incorrect_predictions`), proving that rare misclassifications are strictly confined to minor natural overlaps between *Versicolor* and *Virginica*.
   * Achieved a **97.37% final generalization accuracy** on the hold-out testing set.

---

## Tech Stack Used
* **Language:** Python
* **Libraries:** `scikit-learn`, `pandas`, `numpy`, `seaborn`, `matplotlib`

---

## Visualizations Generated (Available in Current Directory)
* output_1.png: Feature Distribution Histograms
* output_2.png: Feature comparision Pairplot
* output_3.png: Model comparision on accuracy histogram
* output_4.png: Confusion matrices for all models
* output_5.png: Per-Class F1 score by Model
* output_6.png: Random Forest Feature Importance Comparison
* output_7.png: Cross Validation Accuracy comparison
