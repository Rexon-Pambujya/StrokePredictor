# Stroke Prediction Web App

## 🧠 Project Description: Stroke Prediction Web App

This project aims to build a machine learning-powered web application that predicts the probability of stroke in individuals based on health and demographic factors. Using a cleaned and preprocessed healthcare dataset, multiple models were trained and evaluated to identify individuals at high risk.

![Screenshot of a Stroke Risk Predictor web application interface.](https://github.com/Rexon-Pambujya/StrokePredictor/blob/main/images/Screenshot%202025-06-23%20004028.png)

## 🚀 Project Overview

This project builds a machine learning pipeline in Python to predict the risk of stroke based on patient health records. It covers:

1. **Data Preprocessing**: Cleaning, imputation, encoding, and feature engineering.
2. **Exploratory Data Analysis (EDA)**: Summary statistics and visual insights.
3. **Model Training**: Logistic Regression, Random Forest, and XGBoost with class imbalance handling.
4. **Model Evaluation & Selection**: Precision, recall, F1-score, ROC‑AUC, and recall‑optimized threshold tuning.
5. **Deployment**: A user-friendly Streamlit app with probability distributions, personalized insights, and a radar chart.

---

## 📂 Repository Structure

```
├── app.py                  # Streamlit application
├── heart_stroke.ipynb   # Full pipeline script
├── lr_stroke_pipeline.pkl  # Serialized final model & preprocessors
├── README.md               # This documentation
└── healthcare-dataset-stroke-data.csv  # Raw dataset
```

---

## 🛠 Environment Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/stroke-prediction-app.git
   cd stroke-prediction-app
   ```

2. **Create & activate a virtual environment** (optional but recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Data Preprocessing

1. **Load Data**

   ```python
   import pandas as pd
   df = pd.read_csv("healthcare-dataset-stroke-data.csv")
   ```

2. **Initial Cleaning**

   - Drop `id` column
   - Replace `"Unknown"` in `smoking_status` with `NaN`
   - Impute `bmi` using mean
   - Fill missing `smoking_status` with mode

3. **Train‑Validation‑Test Split**

   - 60% train, 20% validation, 20% test using stratified sampling

4. **Feature Pipelines**

   - **Logistic Regression**

     - One‑hot encode categorical features
     - Standardize numeric features

   - **Random Forest & XGBoost**

     - Label‑encode categorical features
     - No scaling
     - Apply SMOTE to address class imbalance

---

## 📈 Exploratory Data Analysis

- **Missing values**: `bmi` had \~5% missing values.
- **Class imbalance**: Only \~4.7% of samples with stroke.
- **Correlations**: Age, glucose, and BMI show positive correlation with stroke.

_Visuals included in Jupyter notebook (not shown here)._

---

## 🤖 Model Training & Evaluation

### Models Trained

- **Logistic Regression** (`class_weight='balanced'`)
- **Random Forest** (`n_estimators=100`, trained after SMOTE)
- **XGBoost** (`n_estimators=100`, `learning_rate=0.1`, `scale_pos_weight`)

### Validation Performance

|                          Model | Recall (stroke) | Precision | ROC‑AUC |
| -----------------------------: | --------------: | --------: | ------: |
| Logistic Regression (Balanced) |            0.80 |      0.12 |    0.84 |
|          Random Forest (SMOTE) |            0.16 |      0.10 |    0.75 |
| XGBoost (SMOTE + scale_weight) |            0.48 |      0.12 |    0.78 |

**Chosen**: Logistic Regression prioritizes **recall** (sensitivity).

---

## 🎯 Threshold Tuning

To achieve **≥ 90% recall**, we searched the precision‑recall curve and selected the **highest threshold** meeting the target:

> This threshold was chosen because it is better to classify a person in a risk category than to miss a patient with a high heart stroke risk.

```python
from sklearn.metrics import precision_recall_curve
prec, rec, thr = precision_recall_curve(y_val, y_proba)
# Pick max threshold where recall ≥ 0.90
target = 0.90
valid = thr[rec[:-1] >= target]
best_thr = valid.max()
print(best_thr)  # ~0.284
```

- **Final threshold**: **0.284**
- At this cutoff: **Recall** = 0.90, **Precision** ≈ 0.097

---

## 💾 Final Pipeline & Model Saving

After tuning, retrain on **train+val** and serialize the pipeline:

```python
import pickle
with open("lr_stroke_pipeline.pkl","wb") as f:
    pickle.dump({
        "model": lr_final,
        "scaler": scaler,
        "onehot": ohe,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "threshold": best_thr
    }, f)
```

---

## 🌐 Streamlit Web App

The `app.py` script implements a Streamlit UI:

1. **Sidebar** inputs for patient features.
2. **Prediction** of stroke vs. no-stroke probabilities.
3. **Horizontal bar chart** of probabilities.
4. **Colored distribution histogram** contrasting patient with population.
5. **Personalized insights** based on deviation from population averages.
6. **Radar chart** comparing patient feature ratios to population baseline.

### Running the App

```bash
python -m streamlit run app.py
```

- Ensure `lr_stroke_pipeline.pkl` and `healthcare-dataset-stroke-data.csv` are in the same folder.
- The app opens in your browser at `http://localhost:8501`.

---

![Streamlit interface](https://github.com/Rexon-Pambujya/StrokePredictor/blob/main/images/Screenshot%202025-06-22%20015104.png)

## 📚 Future Work

- Hyperparameter tuning (GridSearchCV)
- Incorporate more models (SVM, Neural Nets)
- Enhance EDA & visual dashboards
- Add user authentication & logging for clinical use

---

## 🎓 Acknowledgments

- Dataset: [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

---

_Feel free to open issues or submit pull requests!_
