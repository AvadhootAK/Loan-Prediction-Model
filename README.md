
# Loan Approval Prediction using Machine Learning

This project predicts whether a loan application will be approved based on applicant details such as income, education, credit history, and property area. The model is deployed using Streamlit for interactive prediction.

## Project Overview

The goal of this project was to build a supervised machine learning model that can classify loan approval status. The dataset was preprocessed, encoded, scaled, and then used to train and evaluate multiple models. Logistic Regression was finalized based on accuracy and interpretability.

## Tech Stack

* Python
* Pandas, NumPy, Scikit-learn
* Streamlit
* Joblib

## Steps Involved

1. **Data Preprocessing**

   * Handled missing values
   * Encoded categorical features using Label Encoding
   * Scaled numerical features using StandardScaler

2. **Exploratory Data Analysis (EDA)**

   * Checked distributions and relationships between features
   * Observed that `Credit_History` was the most influential feature

3. **Model Building**

   * Tested Logistic Regression, Decision Tree, and Random Forest
   * Selected Logistic Regression for balanced accuracy and simplicity
   * Final model achieved around 86% accuracy

4. **Model Deployment**

   * Built a Streamlit web app for real-time predictions
   * User inputs are encoded and scaled before prediction
   * Model predicts whether a loan will be approved or not

## Files Included

* `loan_prediction.ipynb` – Jupyter notebook containing data preprocessing, training, and evaluation
* `app.py` – Streamlit app script for model deployment
* `loan_model.joblib` – Saved trained model
* `scaler.joblib` – Scaler used for numeric feature standardization
* `requirements.txt` – List of dependencies required to run the project

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/loan-prediction-ml.git
   ```
2. Navigate to the project directory:

   ```bash
   cd loan-prediction-ml
   ```
3. Install dependencies:

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Output

* The app takes user inputs such as gender, income, dependents, credit history, etc.
* After processing, it displays whether the **loan is approved or not approved**.

## Insights

* Credit History plays a dominant role in loan approval prediction.
* The dataset is somewhat imbalanced, which affects model generalization.
* Future improvement can include using more balanced data and testing models like Random Forest, XGBoost, or LightGBM.

## Author

Avadhoot Kulkarni
Electronics and Telecommunication Engineer | Data Science & Machine Learning Enthusiast


