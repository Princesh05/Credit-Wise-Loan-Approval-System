# 🏦 Credit Wise Loan System

An end-to-end Machine Learning based Loan Approval Prediction System that predicts whether a loan application is likely to be **Approved** or **Not Approved** based on applicant and financial information.

The project uses a trained **Gaussian Naive Bayes** classification model and provides an interactive **Streamlit web application** for making predictions.

---

## 📌 Project Overview

The Credit Wise Loan System is designed to automate the initial loan approval prediction process.

The user provides information such as:

- Age
- Gender
- Marital Status
- Annual Income
- Loan Amount
- Credit Score
- Number of Dependents
- Existing Loans
- Employment Status

The trained Machine Learning model processes these inputs and predicts the loan status.

### Prediction Classes

| Value | Meaning |
|------:|---------|
| `0` | ❌ Loan Not Approved |
| `1` | ✅ Loan Approved |

---

## 🎯 Objectives

The main objectives of this project are:

- To analyze loan applicant data.
- To preprocess and transform applicant information.
- To perform feature engineering.
- To train a Machine Learning classification model.
- To predict loan approval status.
- To deploy the trained model using Streamlit.
- To provide a simple and user-friendly interface for loan prediction.

---

## 🧠 Machine Learning Model

The project uses:

**Gaussian Naive Bayes (GaussianNB)**

The model was trained using the processed training dataset and evaluated on test data.

The trained model, scaler, and feature information are saved together using `joblib`.

```python
loan_artifact = {
    "model": NB_model,
    "scaler": scaler,
    "feature_names": list(x.columns)
}
