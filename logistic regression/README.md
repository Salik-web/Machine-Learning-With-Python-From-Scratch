# Logistic Regression

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is Logistic Regression?

Despite having "regression" in the name, **Logistic Regression is a classification algorithm** — it predicts which category something belongs to, not a numeric value.

It answers questions like:
- Will this employee leave the company? (**Yes / No**)
- Is this email spam? (**Yes / No**)
- Will this patient get the disease? (**Yes / No**)

### How it works

Linear Regression predicts a number. Logistic Regression takes that number and passes it through the **Sigmoid function**, which squeezes any value into a range between 0 and 1.

```
Output = 1 / (1 + e^(-x))
```

The output is then interpreted as a **probability**:
- Output > 0.5 → predict class **1** (Yes)
- Output < 0.5 → predict class **0** (No)

### Linear vs Logistic Regression

| | Linear Regression | Logistic Regression |
|--|-------------------|---------------------|
| **Output** | Any number | 0 or 1 (probability) |
| **Use case** | Predict salary, price | Predict yes/no outcomes |
| **Function** | Straight line | S-shaped sigmoid curve |

---

## About This Notebook

In this notebook we predict whether an **employee will leave the company or stay**, based on HR data with 14,999 employees.

### Dataset (`Logistic_reg_ex.csv`)

| Column | Description |
|--------|-------------|
| `satisfaction_level` | Employee satisfaction (0 to 1) |
| `last_evaluation` | Last performance evaluation score |
| `number_project` | Number of projects assigned |
| `average_montly_hours` | Average monthly working hours |
| `time_spend_company` | Years at the company |
| `Work_accident` | Had a work accident (0/1) |
| `left` | **Target** — did the employee leave? (1=Yes, 0=No) |
| `promotion_last_5years` | Got promoted in last 5 years (0/1) |
| `Department` | Department name |
| `salary` | Salary level (low, medium, high) |

---

## What the Code Does

### Step 1 — Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### Step 2 — Exploratory Data Analysis (EDA)

Before training, we explore the data to understand which features impact retention.

**Check how many employees left vs stayed:**
```python
left = df[df['left'] == 1]
retained = df[df['left'] == 0]
print(left.shape, retained.shape)
```

**Compare average values between both groups:**
```python
df.groupby(['left']).mean(numeric_only=True)
```
This reveals that employees who left had lower satisfaction and worked more hours.

**Visualize salary vs retention:**
```python
pd.crosstab(df['salary'], df['left']).plot(kind='bar')
```
Low salary employees leave the most — a clear pattern.

**Visualize department vs retention:**
```python
pd.crosstab(df['Department'], df['left']).plot(kind='bar')
```
Sales and support departments have the highest attrition.

### Step 3 — Feature Selection
```python
subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years']]
```
We select the features with the most direct impact on retention based on EDA.

### Step 4 — Handle Categorical Data (salary)

The `salary` column has text values (low, medium, high) — we convert them to numbers using One Hot Encoding:
```python
subdf_dummies = pd.get_dummies(df['salary'], dtype=int, prefix='Salary')
df_with_dummies = pd.concat([subdf, subdf_dummies], axis='columns')
```

This creates three new columns: `Salary_high`, `Salary_low`, `Salary_medium`.

### Step 5 — Prepare Features and Target
```python
x = df_with_dummies   # input features
y = df['left']        # target (0 = stayed, 1 = left)
```

### Step 6 — Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```
80% of data is used for training, 20% for testing.

### Step 7 — Train the Model
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

### Step 8 — Make Predictions
```python
model.predict(X_test.head())
```
Returns an array of 0s and 1s — 0 means the model predicts the employee stays, 1 means they leave.

### Step 9 — Evaluate the Model
```python
model.score(X_test, y_test)
```
Returns accuracy — how many predictions were correct out of all test samples.

---

## Key Takeaways

- Logistic Regression is used for **classification**, not regression — despite the name
- The **Sigmoid function** converts any number into a probability between 0 and 1
- Always do **EDA first** — understanding the data before training leads to better feature selection
- **One Hot Encoding** is required for categorical text columns like salary
- `test_size=0.2` means 20% of data is held back for testing — the model never sees it during training

---

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
```

Install with:
```bash
pip install pandas numpy matplotlib scikit-learn
```

---

*This is part of the Machine Learning With Python From Scratch series. More topics coming soon!*
