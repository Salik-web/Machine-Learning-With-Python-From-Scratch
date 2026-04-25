# One Hot Encoding

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is One Hot Encoding?

Machine learning models work with numbers — they cannot understand text like "BMW X5" or "Audi A5". **One Hot Encoding** is a technique to convert categorical (text) data into numbers that the model can process.

For example, if you have 3 car brands:

| Car Model |
|-----------|
| BMW X5 |
| Audi A5 |
| Mercedez Benz C class |

One Hot Encoding converts this into:

| Audi A5 | BMW X5 | Mercedez Benz C class |
|---------|--------|----------------------|
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 0 | 0 | 1 |

Each category gets its own column with a `1` where it applies and `0` everywhere else.

### Why not just use 1, 2, 3?
If you assigned numbers like Audi=1, BMW=2, Mercedez=3, the model would think Mercedez is "greater than" Audi — which makes no sense for categories. One Hot Encoding avoids this problem.

### Dummy Variable Trap
When you have 3 categories, you only need **2 columns** — not 3. If Audi=0 and BMW=0, it must be Mercedez. Having all 3 causes multicollinearity (redundant information). Always drop one column.

---

## About This Notebook

This notebook covers **two ways** to do One Hot Encoding and then trains a Linear Regression model to predict car sell prices.

### Dataset (`one_hot_encoder_ex.csv`)

| Column | Description |
|--------|-------------|
| `Car Model` | Car brand (BMW X5, Audi A5, Mercedez Benz C class) |
| `Mileage` | Miles driven |
| `Sell Price($)` | The price we want to predict |
| `Age(yrs)` | Age of the car in years |

---

## Method 1 — Pandas `get_dummies`

### Step 1 — Create Dummy Columns
```python
dummies = pd.get_dummies(df['Car Model'], dtype=int)
```
Creates a separate column for each car brand with 1s and 0s. `dtype=int` ensures the output is `1/0` instead of `True/False`.

### Step 2 — Merge with Original DataFrame
```python
merged = pd.concat([df, dummies], axis='columns')
```
Combines the original data with the new dummy columns side by side.

### Step 3 — Drop the Original Text Column
```python
final = merged.drop(['Car Model'], axis='columns')
```
Remove the original `Car Model` column since we now have numeric columns for it.

### Step 4 — Drop One Dummy Column (Avoid Dummy Variable Trap)
```python
final = final.drop(['Mercedez Benz C class'], axis='columns')
```
Drop one category to avoid redundancy. If both Audi and BMW columns are 0, the model knows it's Mercedez.

### Step 5 — Train the Model
```python
x = final.drop(['Sell Price($)'], axis='columns')
y = final['Sell Price($)']

reg = linear_model.LinearRegression()
reg.fit(x, y)
```

### Step 6 — Predict
```python
# Column order: [Mileage, Age, Audi A5, BMW X5]
reg.predict([[45000, 4, 0, 0]])  # Mercedez, 45000 miles, 4 years old
reg.predict([[86000, 7, 0, 1]])  # BMW X5, 86000 miles, 7 years old
```

---

## Method 2 — Sklearn `OneHotEncoder` with `ColumnTransformer`

A more production-ready approach using sklearn's built-in tools.

### Step 1 — Label Encode First
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dfle['Car Model'] = le.fit_transform(dfle['Car Model'])
```
Converts car names to numbers (Audi=0, BMW=1, Mercedez=2) as a first step.

### Step 2 — Extract Features as Array
```python
a = dfle[['Car Model', 'Mileage', 'Age(yrs)']].values
b = dfle['Sell Price($)'].values
```

### Step 3 — Apply OneHotEncoder via ColumnTransformer
```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('Car Model', OneHotEncoder(), [0])], remainder='passthrough')
a = ct.fit_transform(a)
```
- `[0]` = apply OHE to column index 0 (Car Model)
- `remainder='passthrough'` = keep all other columns unchanged
- After this, column order becomes: `[Audi A5, BMW X5, Mercedez, Mileage, Age]`

### Step 4 — Drop First Column (Avoid Dummy Variable Trap)
```python
a = a[:, 1:]
```
Drops the first OHE column. Remaining order: `[BMW X5, Mercedez, Mileage, Age]`

### Step 5 — Train and Predict
```python
reg2 = linear_model.LinearRegression()
reg2.fit(a, b)

# Column order: [BMW X5, Mercedez Benz, Mileage, Age]
reg2.predict([[0, 1, 45000, 4]])  # Mercedez, 45000 miles, 4 years old
```

---

## Key Takeaways

- Machine learning models need numbers — use One Hot Encoding for categorical data
- Always drop one dummy column to avoid the dummy variable trap
- `pd.get_dummies()` is quick and simple for exploration
- `sklearn OneHotEncoder` with `ColumnTransformer` is better for production pipelines
- After `ColumnTransformer`, the encoded columns come **first** — always check the column order before predicting

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
