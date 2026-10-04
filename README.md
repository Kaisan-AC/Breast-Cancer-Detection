# Breast Cancer Detection using Machine Learning

A machine learning project that classifies breast tumors as **malignant** or **benign** using a Random Forest classifier trained on the Wisconsin Breast Cancer dataset from scikit-learn.

> **Disclaimer:** This project is for educational purposes only. It is not a medical device and must not be used for real diagnosis or treatment decisions. Always consult a qualified healthcare professional.

---

## Results

The model was evaluated on a held-out test set (20% of the data, 114 samples).

| Metric | Value |
|---|---|
| Training Accuracy | 100.00% |
| Test Accuracy | **95.61%** |

**Classification report (test set):**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Malignant | 0.95 | 0.93 | 0.94 | 42 |
| Benign | 0.96 | 0.97 | 0.97 | 72 |
| **Overall accuracy** | | | **0.96** | 114 |

> The gap between training accuracy (100%) and test accuracy (95.61%) suggests the model slightly overfits the training data. See [Future Improvements](#future-improvements).

---

## Dataset

- **Source:** [Breast Cancer Wisconsin (Diagnostic) Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), loaded via `sklearn.datasets.load_breast_cancer`
- **Samples:** 569
- **Features:** 30 numeric features computed from digitized images of fine needle aspirate (FNA) of breast masses (radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension, and their mean / standard error / worst values)
- **Target:** `0` = malignant, `1` = benign

---

## Approach

1. Load the dataset with scikit-learn.
2. Split into train (80%) and test (20%) sets using a stratified split (`random_state=42`) to preserve class balance.
3. Standardize features with `StandardScaler` (fit on the training set only to avoid data leakage).
4. Train a `RandomForestClassifier` (`random_state=42`).
5. Evaluate with accuracy and a full classification report.
6. Save the trained model and scaler with `joblib`.

---

## Repository Structure

```
├── Cancer_detection.ipynb     # Training and evaluation notebook
├── breast_cancer_model.pkl    # Trained Random Forest model
├── scaler.pkl                 # Fitted StandardScaler
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install scikit-learn pandas joblib jupyter
```

> Use the same scikit-learn version the model was trained with (or retrain via the notebook) to avoid pickle compatibility issues.

---

## Usage

### Retrain the model

Open and run the notebook:

```bash
jupyter notebook Cancer_detection.ipynb
```

### Make predictions with the saved model

```python
import joblib
from sklearn.datasets import load_breast_cancer

model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example: use a few samples from the dataset
data = load_breast_cancer(as_frame=True)
samples = data.data.iloc[:5]

# Scale first, then predict
samples_scaled = scaler.transform(samples)
predictions = model.predict(samples_scaled)

for pred in predictions:
    print("Benign" if pred == 1 else "Malignant")
```

Remember that new input must contain the same 30 features, in the same order, as the training data.

---

## Tech Stack

- Python 3.10
- scikit-learn
- pandas
- joblib
- Jupyter Notebook

---

## Future Improvements

- Hyperparameter tuning (e.g., `GridSearchCV` / `RandomizedSearchCV`) to reduce overfitting
- Cross-validation for more reliable performance estimates
- Compare with other models (Logistic Regression, SVM, Gradient Boosting, XGBoost)
- Feature importance analysis and visualization
- Confusion matrix and ROC curve
- Build a simple web app (Streamlit / Flask) for interactive predictions
- Put scaler and model into a single scikit-learn `Pipeline`

---

## License

This project is licensed under the MIT License. Feel free to use and modify it.

---

## Author

Kaisan-AC
GitHub: https://github.com/Kaisan-AC
