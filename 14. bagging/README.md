# Bagging

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is Bagging?

**Bagging** (Bootstrap Aggregating) is an ensemble method that trains multiple copies of the same model on different random subsets of the training data and combines their predictions.

### How it works

1. Randomly sample the training data **with replacement** (bootstrap) — some rows appear multiple times, some not at all
2. Train a separate model on each sample
3. Combine predictions by majority vote (classification) or average (regression)

```
Training data (100 rows)
      ↓
Sample 1 (random 100 rows with replacement) → Model 1 → prediction
Sample 2 (random 100 rows with replacement) → Model 2 → prediction
Sample 3 (random 100 rows with replacement) → Model 3 → prediction
                                                    ↓
                                              Majority vote
```

### Why does it help?

A single model has high variance — small changes in training data produce very different models. Bagging reduces variance by averaging many models trained on slightly different data.

### OOB Score (Out-of-Bag)

When bagging samples rows with replacement, some rows are never picked for a given model. These are **out-of-bag** samples. Bagging can test each model on its own OOB rows — giving you a free accuracy estimate without a separate test set.

### Random Forest vs BaggingClassifier

Random Forest is essentially Bagging with Decision Trees plus an extra trick — at each split it only considers a random subset of features. This makes trees even more independent from each other.

---

## About This Notebook

Predicts **heart disease** (yes/no) using the heart disease dataset. Compares standalone models vs bagging vs Random Forest.

### Dataset (`bagging_ex.csv`)

| Column | Description |
|--------|-------------|
| `Age` | Patient age |
| `Sex` | Gender |
| `ChestPainType` | Type of chest pain |
| `RestingBP` | Resting blood pressure |
| `Cholesterol` | Serum cholesterol |
| `FastingBS` | Fasting blood sugar |
| `RestingECG` | Resting ECG results |
| `MaxHR` | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina |
| `Oldpeak` | ST depression |
| `ST_Slope` | Slope of peak exercise ST segment |
| `HeartDisease` | **Target** — 1=disease, 0=no disease |

---

## What the Code Does

### Step 1 — Load and Explore
```python
df = pd.read_csv('bagging_ex.csv')
df.isna().sum()
df['HeartDisease'].value_counts()
```

### Step 2 — Remove Outliers (Z-Score)
```python
from scipy import stats
df_numeric = df.select_dtypes(include=[np.number])
df_clean = df[(np.abs(stats.zscore(df_numeric)) < 3).all(axis=1)]
```

### Step 3 — Encode Categorical Columns
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_clean = df_clean.copy()
df_clean[col_to_numeric] = df_clean[col_to_numeric].apply(le.fit_transform)
```

### Step 4 — Scale Features
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
```
StandardScaler brings all features to mean=0, std=1 — important for distance-based models.

### Step 5 — Standalone Models with Cross Validation
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(DecisionTreeClassifier(), x, y, cv=5)
scores1 = cross_val_score(SVC(), x, y, cv=5)
```
Baseline scores without bagging.

### Step 6 — BaggingClassifier with SVC
```python
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(estimator=SVC(), n_estimators=100, oob_score=True, random_state=42)
```

### Step 7 — BaggingClassifier with Decision Tree
```python
bagging2 = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
```

### Step 8 — Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
scores3 = cross_val_score(RandomForestClassifier(), x, y, cv=5)
```
Compare Random Forest performance vs manual bagging.

---

## Key Takeaways

- Bagging trains many models on random bootstrap samples and averages predictions
- Reduces variance — a single Decision Tree or SVM can be unstable; bagging stabilises it
- `oob_score=True` gives a free evaluation without using the test set
- Random Forest is bagging + random feature selection — usually more powerful
- `n_estimators=100` is a good default; more trees = more stable, slower training

---

## Requirements

```
pandas
numpy
scipy
scikit-learn
```

Install with:
```bash
pip install pandas numpy scipy scikit-learn
```

---

*This is part of the Machine Learning With Python From Scratch series. More topics coming soon!*
