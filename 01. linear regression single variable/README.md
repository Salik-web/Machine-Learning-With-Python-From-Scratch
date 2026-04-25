# Linear Regression - Single Variable

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is Linear Regression?

Linear Regression is one of the simplest and most widely used machine learning algorithms. It is used to **predict a continuous numeric value** based on one or more input variables.

The idea is simple: find the best straight line that fits your data.

The equation of that line is:

```
y = mx + b
```

Where:
- `y` = predicted value (what we want to find)
- `x` = input feature
- `m` = slope (how much y changes when x increases by 1)
- `b` = intercept (value of y when x is 0)

The algorithm learns the values of `m` and `b` from the training data.

---

## About This Notebook

In this notebook, we predict **per capita income** based on **year** using a real-world dataset.

### Dataset (`linear_regression_Ex.csv`)
| Column | Description |
|--------|-------------|
| `year` | The year (1970 to 2016) |
| `per capita income (US$)` | Average income per person in that year |

### What the Code Does

**Step 1 — Import Libraries**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
```
- `pandas` — for loading and handling data
- `numpy` — for numerical operations
- `matplotlib` — for plotting graphs
- `sklearn` — for building the linear regression model

**Step 2 — Load the Data**
```python
df = pd.read_csv('linear_regression_Ex.csv')
```
Loads the CSV file into a pandas DataFrame.

**Step 3 — Visualize the Data**
```python
plt.scatter(df['year'], df['per capita income (US$)'], color='red', marker='.')
plt.show()
```
Scatter plot to see the relationship between year and income. You'll notice income clearly increases over time — a linear pattern.

**Step 4 — Train the Model**
```python
reg = linear_model.LinearRegression()
reg.fit(df[['year']], df['per capita income (US$)'])
```
- `LinearRegression()` creates the model
- `.fit()` trains it on our data — it finds the best values for slope and intercept
- Note: `df[['year']]` uses double brackets because sklearn needs a 2D input

**Step 5 — Check Model Accuracy**
```python
reg.score(df[['year']], df['per capita income (US$)'])
```
Returns the **R² score** — how well the model fits the data.
- Score close to `1.0` = great fit
- Our model scores ~**0.98** which means 98% of the income variation is explained by year

**Step 6 — Make Predictions**
```python
reg.predict([[2020]])
```
Predicts the per capita income for the year 2020.

---

## Key Takeaways

- Linear Regression finds a straight line relationship between input and output
- Always visualize your data before training
- Use `df[['column']]` (double brackets) for sklearn — it needs 2D input
- R² score tells you how good your model is

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
