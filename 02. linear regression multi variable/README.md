# Linear Regression - Multiple Variables

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is Multiple Variable Linear Regression?

In single variable linear regression, we predict an output using **one input**. But in the real world, outcomes depend on **multiple factors**.

For example, a person's salary depends on:
- Years of experience
- Test score
- Interview score

Multiple Linear Regression handles exactly this — it finds the best relationship between **several input variables** and one output.

The equation becomes:

```
y = m1*x1 + m2*x2 + m3*x3 + b
```

Where each `x` is a different input feature and each `m` is its corresponding weight (how much it impacts the output).

---

## About This Notebook

In this notebook, we predict an employee's **salary** based on their experience, test score, and interview score.

### Dataset (`Linear_Reg_multi_ex.csv`)

| Column | Description |
|--------|-------------|
| `experience` | Years of experience (stored as words e.g. "five") |
| `test_score(out of 10)` | Candidate's test score |
| `interview_score(out of 10)` | Candidate's interview score |
| `salary($)` | The salary we want to predict |

---

## What the Code Does

### Step 1 — Import Libraries
```python
import pandas as pd
import numpy as np
from sklearn import linear_model
```

### Step 2 — Load & Inspect Data
```python
df = pd.read_csv('Linear_Reg_multi_ex.csv')
```
The dataset has some issues we need to fix before training:
- `experience` column is stored as **words** (e.g. "five", "seven")
- Some values in `experience` and `test_score` are **missing (NaN)**

### Step 3 — Convert Experience Words to Numbers
```python
word_to_num = {'one': 1, 'two': 2, 'three': 3, ...}
df['experience'] = df['experience'].map(word_to_num)
```
`.map()` replaces each word with its numeric equivalent. Any word not in the dictionary becomes `NaN`.

### Step 4 — Handle Missing Values
```python
df['experience'] = df['experience'].fillna(0)
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].median())
```
- Missing experience is filled with `0` (no experience)
- Missing test score is filled with the **median** of existing scores

### Step 5 — Train the Model
```python
reg = linear_model.LinearRegression()
reg.fit(df.drop(['salary($)'], axis='columns'), df['salary($)'])
```
- Drop the salary column to get the input features (X)
- Train the model to predict salary (y)

### Step 6 — Make Predictions
```python
reg.predict([[2, 9, 6]])
```
Predicts salary for someone with 2 years experience, test score 9, interview score 6.

### Step 7 — Check Model Accuracy
```python
reg.score(df.drop(['salary($)'], axis='columns'), df['salary($)'])
```
Returns the R² score. Note: this dataset only has 8 rows so the score is around 40% — with more data, it would be much higher.

---

## Saving Your Model with Pickle

After training a model, you don't want to retrain it every time you use it. **Pickle** lets you save the trained model to a file and load it later.

### Save the Model
```python
import pickle

with open('model_pickle', 'wb') as file:
    pickle.dump(reg, file)
```
- `'wb'` = write in binary mode
- `pickle.dump()` serializes the model and saves it to the file

### Load the Model
```python
with open('model_pickle', 'rb') as file:
    mp = pickle.load(file)
```
- `'rb'` = read in binary mode
- `pickle.load()` deserializes and restores the saved model

### Use the Loaded Model
```python
mp.predict([[12, 10, 10]])
```
The loaded model works exactly like the original — no need to retrain!

### Why Pickle?
- Train once, use forever
- Deploy your model in other apps or scripts
- Save time — especially when training on large datasets takes hours

---

## Key Takeaways

- Multiple regression uses several input features to predict one output
- Always clean your data before training — handle missing values and wrong data types
- Use `axis='columns'` when dropping a column with `.drop()`
- Save your trained model with Pickle so you don't have to retrain it every time

---

## Requirements

```
pandas
numpy
scikit-learn
```

Install with:
```bash
pip install pandas numpy scikit-learn
```

---

*This is part of the Machine Learning With Python From Scratch series. More topics coming soon!*
