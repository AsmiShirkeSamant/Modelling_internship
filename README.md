# Iris Species Classification Project

Basic machine learning project focused on exploring, visualizing, and classifying the classic **Iris Flower Dataset**. The goal is to accurately predict the species of an iris flower based on its sepal and petal measurements.

---

## Project Status:
I have successfully set up the core workflow pipeline and established a strong baseline using **Logistic Regression**. The project is now ready for experimentation with more complex algorithms.

## Completed Workflow:

1. **Data Acquisition & Preprocessing**
   * Loaded the dataset via `scikit-learn` / `seaborn`.
   * Inspected features (`sepal length`, `sepal width`, `petal length`, `petal width`).
   * Split the data into training (80%) and testing (20%) sets to ensure valid evaluation.

2. **Baseline Model Implementation**
   * Trained a Logistic Regression model.
   * Achieved high baseline accuracy on the test set, demonstrating that the features provide strong predictive signals.

---

## Tech Stack Used
* Language: Python
* Libraries: `scikit-learn`, `pandas`, `numpy`, `seaborn`, `matplotlib`

---

## Next Steps
I will move on to implement the following classification models to see how their decision boundaries differ:

- [ ] K-Nearest Neighbors (KNN): To explore instance-based spatial clustering.
- [ ] Decision Trees: To build an interpretable, rule-based classification map.
- [ ] Support Vector Machines (SVM): To find optimal high-dimensional class boundaries.
- [ ] Random Forest: To look at ensemble methods and prevent overfitting.
- [ ] Model Evaluation: Compare all models side-by-side using Confusion Matrices and Classification Reports ($F_1$-score, Precision, Recall).
