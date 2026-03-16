# 🚢 Titanic Survival Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-green.svg)

A machine learning project that predicts whether a passenger survived the Titanic disaster using **Logistic Regression** classification.

---

## 📁 Project Structure

```
titanic_project/
│
├── data/
│   └── titanic.csv              # Dataset file
│
├── titanic_logistic.py          # Main Python script
├── titanic_results.png          # Output chart (generated after running)
├── README.md                    # Project documentation
```

---

## 📌 Problem Statement

> **Can we predict whether a Titanic passenger survived based on features like gender, age, passenger class, and fare?**

This is a **binary classification** problem:
- `0` → Passenger Died
- `1` → Passenger Survived

---

## 📊 Dataset

| Column | Description |
|--------|-------------|
| `survived` | Target variable: 0 = Died, 1 = Survived |
| `pclass` | Ticket class (1 = First, 2 = Second, 3 = Third) |
| `sex` | Gender (male / female) |
| `age` | Age in years (has missing values) |
| `sibsp` | Number of siblings / spouses aboard |
| `parch` | Number of parents / children aboard |
| `fare` | Ticket fare paid |
| `embarked` | Port of embarkation (S / C / Q) |

- **Total Rows:** 891 passengers
- **Missing Values:** `age` column has ~20% missing values

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.x | Programming language |
| VS Code | Code editor / IDE |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | ML model, pipeline, evaluation |
| Matplotlib | Visualization |
| Seaborn | Heatmap and plots |

---

## ⚙️ Installation & Setup

### Step 1 — Install Python
Download from: https://python.org

### Step 2 — Install Required Libraries
Open terminal in VS Code (`Ctrl + `` ` ``) and run:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Step 3 — Clone / Download Project
Place all files in a folder:
```
my_project/
├── data/
│   └── titanic.csv
└── titanic_logistic.py
```

### Step 4 — Run the Code
```bash
python titanic_logistic.py
```

Or press **F5** in VS Code.

---

## 🔄 Workflow

```
Load Dataset
     ↓
Explore Data (shape, missing values, class counts)
     ↓
Clean Data (fill missing, encode categorical)
     ↓
Split Data (80% train / 20% test)
     ↓
Build Pipeline (StandardScaler + LogisticRegression)
     ↓
Train Model
     ↓
Evaluate (Classification Report, Confusion Matrix, ROC-AUC)
     ↓
Cross Validation (5-Fold)
     ↓
Feature Importance (Coefficients)
     ↓
Predict New Passenger
```

---

## 🧪 Model Details

### Algorithm
**Logistic Regression** — a linear classification algorithm that predicts the probability of a binary outcome.

### Pipeline Steps
```python
Pipeline([
    ('scaler', StandardScaler()),         # Step 1: Normalize features
    ('model', LogisticRegression(         # Step 2: Train classifier
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ))
])
```

### Why StandardScaler?
Logistic Regression is sensitive to feature scale. StandardScaler ensures all features have:
- Mean = 0
- Standard Deviation = 1

### Why class_weight='balanced'?
The dataset has slightly more deaths than survivals. `balanced` adjusts weights automatically to handle this imbalance.

---

## 📈 Results

| Metric | Died (0) | Survived (1) |
|--------|----------|--------------|
| Precision | 0.83 | 0.76 |
| Recall | 0.84 | 0.75 |
| F1-Score | 0.84 | 0.75 |

| Overall Metric | Score |
|----------------|-------|
| Accuracy | ~80% |
| ROC-AUC Score | ~0.87 |
| CV F1 (5-Fold) | ~0.80 ± 0.03 |

---

## 🔍 Feature Importance

Features ranked by impact on survival prediction:

```
sex          →  Strongest predictor (female = higher survival)
pclass       →  First class passengers survived more
fare         →  Higher fare = higher survival
age          →  Younger passengers had slight advantage
embarked     →  Minor effect
sibsp/parch  →  Small families had better outcomes
```

---

## 🔮 Predict a New Passenger

```python
new_passenger = pd.DataFrame({
    'pclass':   [3],
    'sex':      [0],    # 0 = male, 1 = female
    'age':      [22],
    'sibsp':    [0],
    'parch':    [0],
    'fare':     [7.25],
    'embarked': [0]     # 0 = S, 1 = C, 2 = Q
})

prediction = pipeline.predict(new_passenger)
probability = pipeline.predict_proba(new_passenger)[0][1]

print(f"Prediction: {'Survived ✅' if prediction[0] == 1 else 'Died ❌'}")
print(f"Survival Probability: {probability:.2%}")
```

---

## ⚠️ Common Issues & Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip install <module_name>` |
| `ConvergenceWarning` | Increase `max_iter=2000` |
| `FileNotFoundError` | Check that `titanic.csv` is in the `data/` folder |
| Low accuracy | Try `class_weight='balanced'` or use Random Forest |

---

## 🚀 Future Improvements

- [ ] Try Random Forest and compare accuracy
- [ ] Try XGBoost for better performance
- [ ] Add more visualizations (ROC curve, survival by gender chart)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Deploy model as a web app using Flask or Streamlit

---

## 📚 References

- Scikit-learn Documentation: https://scikit-learn.org
- Pandas Documentation: https://pandas.pydata.org
- Titanic Dataset: Kaggle (https://www.kaggle.com/c/titanic)

---

## 👨‍💻 Author

**Data Science Project**
Titanic Survival Prediction Using Machine Learning
March 2026
