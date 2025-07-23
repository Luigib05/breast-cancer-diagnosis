# Breast Cancer Diagnosis Prediction

This project applies machine learning to predict whether a tumor is malignant or benign based on clinical features extracted from digitized images of fine needle aspirate (FNA) of breast masses.

It uses the **Wisconsin Breast Cancer Diagnostic (WBCD)** dataset and follows a structured pipeline including data loading, preprocessing, exploratory data analysis (EDA), model training, and evaluation.

---

## Dataset

- **Source**: [UCI Machine Learning Repository – WDBC Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Samples**: 569
- **Features**: 30 numeric features computed from digitized images of cell nuclei
- **Target variable**: Diagnosis (`M` = malignant, `B` = benign)

---

## Tech Stack

- **Language**: Python 3.11
- **Libraries**:
  - `pandas`, `numpy` for data manipulation
  - `matplotlib`, `seaborn` for data visualization
  - `scikit-learn` for machine learning models

---

## Project Structure

Breast_cancer_project/
│
├── app.py                       
├── models/
│   └── logistic_regression_model.pkl
├── data/
│   └── wdbc.data
├── src/
│   ├── model_training.py
│   ├── preprocessing.py
│   ├── save_model.py
│   └── ...
├── tests/
│   ├── test_load_data.py
│   └── test_model_training.py
├── requirements.txt
└── README.md

## Steps Completed

### 1. Exploratory Data Analysis (EDA)
- Verified data completeness (no missing values)
- Examined class distribution (benign vs malignant)
- Visualized feature distributions by class
- Computed and visualized feature correlations
- Identified top predictors with |r| > 0.4

### 2. Data Preprocessing
- Selected 10 most correlated features
- Applied feature scaling with `StandardScaler`

### 3. Model Training & Evaluation
- **Logistic Regression** (baseline):  
  Accuracy: **96.5%**  
  F1-score: **95.2%**

- **Random Forest (Tuned)**:  
  Accuracy: 95.9%  
  F1-score: 94.3%

- Final model selected: **Logistic Regression** (better recall & simplicity)

### 4. Model Saving & Metrics Logging
- Model saved as 'logistic_regression_model.pkl' using `joblib`
- Evaluation metrics stored in `evaluation_metrics.txt`

### 5. Deployment (Local)
- Built an MVP interactive web app using **Streamlit**
- Allows user input for 10 features via sliders
- Displays model prediction and probability

## 🚀 Run the App Locally

1. Create and activate virtual environment
2. Install dependencies:
   bash
   pip install -r requirements.txt

## Run the app

streamlit run app.py

## Test

To ensure that key scripts (data loading, training) work correctly.
to run: 
 bash
 python -m unittest test/test_load_data.py

## License:

This project is for educational and research purposes only

