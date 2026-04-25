# L1 and L2 Regularization

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is Regularization?

When a Linear Regression model has many features, it can overfit — learning the noise in training data and performing poorly on new data. **Regularization** adds a penalty to the model for having large coefficients, forcing it to keep weights small and generalise better.

### L1 Regularization — Lasso

Lasso adds the **sum of absolute values** of coefficients as a penalty. This can shrink some coefficients all the way to **zero**, effectively removing those features. Lasso does automatic feature selection.

```
Loss = MSE + alpha × Σ|coefficient|
```

### L2 Regularization — Ridge

Ridge adds the **sum of squared values** of coefficients. It shrinks all coefficients but rarely makes them exactly zero — all features stay in the model but with smaller weights.

```
Loss = MSE + alpha × Σ(coefficient²)
```

### alpha Parameter

Controls how strong the regularization penalty is:
- **High alpha** → stronger penalty → smaller coefficients → simpler model → may underfit
- **Low alpha** → weaker penalty → closer to plain Linear Regression → may overfit
- **alpha=0** → no regularization = plain Linear Regression

| | Lasso (L1) | Ridge (L2) |
|--|-----------|-----------|
| **Penalty** | Sum of absolute values | Sum of squared values |
| **Feature selection** | Yes (zeros out features) | No (keeps all features) |
| **Use when** | Many irrelevant features | All features are relevant |

---

## About This Notebook

Predicts **Melbourne property prices** using a dataset with many categorical and numeric features. Compares plain Linear Regression vs Lasso vs Ridge.

### Dataset (`L1_L2_Ex.csv`)

| Column | Description |
|--------|-------------|
| `Suburb` | Suburb name |
| `Rooms` | Number of rooms |
| `Type` | Property type |
| `Method` | Sale method |
| `Distance` | Distance from CBD |
| `Bedroom2` | Number of bedrooms |
| `Bathroom` | Number of bathrooms |
| `Car` | Car spots |
| `Landsize` | Land size |
| `BuildingArea` | Building area |
| `Price` | **Target** — property price |

---

## What the Code Does

### Step 1 — Select Relevant Columns
```python
cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname',
               'Propertycount', 'Distance', 'CouncilArea', 'Bedroom2',
               'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
dataset = dataset[cols_to_use]
```

### Step 2 — Handle Missing Values
```python
cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)
dataset.dropna(inplace=True)
```
Fill numeric columns with 0, then drop rows that still have NaN (text columns).

### Step 3 — One Hot Encode Categorical Columns
```python
dataset_dummies = pd.get_dummies(dataset, drop_first=True, dtype=int)
```
`drop_first=True` automatically handles the dummy variable trap for all categorical columns at once.

### Step 4 — Train/Test Split
```python
train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.25)
```

### Step 5 — Plain Linear Regression (Baseline)
```python
model = LinearRegression()
model.fit(train_X, train_y)
model.score(test_X, test_y)   # test score
model.score(train_X, train_y) # train score
```
If train score >> test score, the model is overfitting.

### Step 6 — Lasso (L1)
```python
from sklearn import linear_model
L1_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
L1_reg.fit(train_X, train_y)
L1_reg.score(test_X, test_y)
```

### Step 7 — Ridge (L2)
```python
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(train_X, train_y)
ridge_reg.score(test_X, test_y)
```

---

## Key Takeaways

- Use regularization when your model overfits (train score much higher than test score)
- **Lasso** removes irrelevant features by zeroing out their coefficients
- **Ridge** keeps all features but shrinks their weights
- `alpha` controls regularization strength — tune it with Grid Search CV
- `drop_first=True` in `get_dummies` handles the dummy variable trap automatically

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
