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

breast-cancer-project/
│
├── data/ # Dataset files (.data, .names)
├── notebook/ # Jupyter notebooks (EDA, experiments)
├── src/ # Python modules for loading and processing data
├── .venv/ # Virtual environment (excluded)
├── main.py # Entry point to run the pipeline
├── requirements.txt # Dependencies
└── README.md # Project description


---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/breast-cancer-project.git
cd breast-cancer-project

2. Create and activate virtual environment:

python -m venv .venv
.venv\Scripts\activate      # On Windows
source .venv/bin/activate  # On macOS/Linux

3. Install dependencies:

pip install -r requirements.txt

4. Run the pipeline:

python main.py

License:

This project is for educational and research purposes only

