# BestBuyCarPrediction-Machine-Learning

### Overview

This project applies **machine learning classification models** to predict whether a used vehicle is a **“Good Buy” or a “Kick (Bad Buy)”**. The dataset included auction, pricing, and vehicle attributes. The analysis highlights how different ML models perform in predicting high-risk car purchases.

---

### Objectives

- Clean and preprocess auction/vehicle dataset
- Engineer features such as pricing ratios and categorical encodings.
- Build and evaluate multiple ML models: Decision Tree, Logistic Regression, and Neural Network.
- Compare models using **accuracy, ROC-AUC, and overfitting risk**.

---

### Tools & Technologies

- **Python (scikit-learn, pandas, matplotlib)**
- **ML Models**: Decision Tree, Logistic Regression, Neural Network (MLP)
- **GridSearchCV** for hyperparameter tuning

---

### Highlights

- Addressed **class imbalance** (87% Good Buy vs 13% Bad Buy).
- Cleaned missing values with median/mode imputation.
- Feature engineered pricing ratios for stronger predictive power.
- Compared interpretability (Decision Tree, Logistic Regression) vs performance (Neural Network).

---

### Key Results

- **Default Decision Tree** → 100% train accuracy, 82% test accuracy → *overfitting*.
- **Tuned Decision Tree (GridSearchCV)** → 89.6% accuracy, simpler model (49 nodes).
- **Logistic Regression** → 89.4% accuracy, interpretable coefficients.
- **Neural Network (MLP)** → 89.4% accuracy, **best ROC-AUC = 0.769**.

**Final model chosen:** **Neural Network (MLP)** for best balance of predictive performance and generalization

---

### Insights

- Cars with **unknown wheel type** or **low auction prices** are strongly associated with being a “Kick.”
- Market valuation features (retail vs auction clean prices) are critical predictors.
- Demonstrated the trade-off between **model interpretability** and **accuracy** in ML.
