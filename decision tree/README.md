# Decision Tree

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is a Decision Tree?

A Decision Tree is a classification algorithm that makes decisions by asking a series of **yes/no questions** about the data, just like a flowchart.

For example, to predict if a Titanic passenger survived:

```
Is the passenger female?
├── Yes → Did they travel in 1st or 2nd class?
│         ├── Yes → Likely SURVIVED
│         └── No  → May not have survived
└── No  → Is their age < 10?
          ├── Yes → Likely SURVIVED
          └── No  → Likely DID NOT SURVIVE
```

Each question is called a **node**, each answer is a **branch**, and the final answer is a **leaf**.

### How does the tree decide which question to ask first?

It picks the question that **splits the data most cleanly** — using a metric called **Gini Impurity** or **Information Gain**. The goal is to separate classes as clearly as possible at each step.

### Overfitting Problem

If you let the tree grow without limits, it memorizes every training example — including noise. This is called **overfitting**. The model performs great on training data but poorly on unseen data.

**Solution — `max_depth`**: Limit how deep the tree can grow. This forces it to learn general patterns instead of memorizing.

---

## About This Notebook

In this notebook we predict whether a **Titanic passenger survived or not** based on their ticket class, sex, age and fare.

### Dataset (`decision_tree.csv`)

| Column | Description |
|--------|-------------|
| `Pclass` | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Sex` | Gender (male/female) |
| `Age` | Age of the passenger |
| `Fare` | Ticket fare paid |
| `Survived` | **Target** — did they survive? (1=Yes, 0=No) |

---

## What the Code Does

### Step 1 — Load and Select Features
```python
df = pd.read_csv('decision_tree.csv')
x = df[['Pclass', 'Sex', 'Age', 'Fare']]
target = df['Survived']
```
We select 4 features that have the most direct impact on survival.

### Step 2 — Encode Sex Column
```python
from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()
x['sex_n'] = le_sex.fit_transform(x['Sex'])
x_n = x.drop(['Sex'], axis='columns')
```
The `Sex` column has text (male/female) — LabelEncoder converts it to numbers:
- `female` → 0
- `male` → 1

### Step 3 — Handle Missing Age Values
```python
x_n['Age'].isnull().sum()  # check how many are missing
x_n['Age'] = x_n['Age'].fillna(x_n['Age'].median())
```
177 passengers (20%) have no age recorded. We fill them with the **median age** so the model can still use those rows instead of discarding them.

### Step 4 — Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(x_n, target, test_size=0.2, random_state=42)
```
80% of data for training, 20% for testing. `random_state=42` ensures the same split every run.

### Step 5 — Train the Model
```python
model = tree.DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
```
- `max_depth=5` — limits tree depth to prevent overfitting
- Without this, accuracy drops from **80% to 74%** due to overfitting

### Step 6 — Evaluate Accuracy
```python
model.score(X_test, y_test)  # ~80%
```

### Step 7 — Compare Predictions vs Actual
```python
results = X_test.copy()
results['Actual'] = y_test.values
results['Predicted'] = model.predict(X_test)
results['Correct'] = results['Actual'] == results['Predicted']

print('Correct predictions:', results['Correct'].sum())
print('Wrong predictions:', (results['Correct'] == False).sum())
```
This creates a clear table showing each passenger, what actually happened, what the model predicted, and whether it was right or wrong.

---

## Why max_depth Matters

| max_depth | Test Accuracy |
|-----------|--------------|
| None (unlimited) | 74% — overfitting |
| 3 | 80% |
| 4 | 80% |
| **5** | **80.45% — best** |
| 6 | 80% |

Limiting the depth forces the model to learn broad patterns rather than memorizing noise in the training data.

---

## Key Takeaways

- Decision Trees make predictions by asking a sequence of yes/no questions
- They are easy to understand and interpret — you can follow the logic
- **Overfitting is the biggest risk** — always set `max_depth`
- Missing values must be handled before training — sklearn can't handle NaN
- `LabelEncoder` is needed for text columns like Sex
- `random_state` in both the model and train_test_split ensures reproducible results

---

## Requirements

```
pandas
scikit-learn
```

Install with:
```bash
pip install pandas scikit-learn
```

---

*This is part of the Machine Learning With Python From Scratch series. More topics coming soon!*
