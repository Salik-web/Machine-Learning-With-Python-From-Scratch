# Random Forest

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is Random Forest?

Random Forest is an **ensemble method** — it builds many Decision Trees and combines their predictions by majority vote.

Each tree is trained on a **random subset** of the training data (bootstrap sampling) and at each split considers only a **random subset of features**. This randomness makes each tree different, and combining many different trees produces a much more accurate and robust model than any single tree.

```
Training data
    ↓
Tree 1 (random subset) → prediction: 0
Tree 2 (random subset) → prediction: 1
Tree 3 (random subset) → prediction: 1
        ↓
   Majority vote → Final prediction: 1
```

### Why is it better than a single Decision Tree?

A single Decision Tree tends to overfit — it memorises the training data. Random Forest reduces overfitting by averaging many trees that each overfit in different ways, cancelling out the noise.

### `n_estimators`

The number of trees to build. More trees = more stable predictions, but slower training. Usually 100 is a good starting point.

---

## About This Notebook

Classifies **Iris flowers** (Setosa, Versicolor, Virginica) using 4 petal and sepal measurements.

### Dataset (`load_iris`)

| Property | Details |
|----------|---------|
| Samples | 150 flowers |
| Features | 4 (sepal length/width, petal length/width) |
| Classes | 3 (Setosa, Versicolor, Virginica) |

---

## What the Code Does

### Step 1 — Load Dataset
```python
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_names'] = [iris.target_names[i] for i in iris.target]
```

### Step 2 — Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['target', 'flower_names'], axis='columns'),
    df['target'], test_size=0.2, random_state=42
)
```

### Step 3 — Train Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)
model.score(X_test, y_test)
```
`n_estimators=10` builds 10 trees. In practice you'd use 100+.

### Step 4 — Predictions
```python
model.predict(X_test)
```
Each tree votes and the majority class wins.

### Step 5 — Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
sn.heatmap(cm, annot=True)
```

---

## Key Takeaways

- Random Forest builds many trees on random subsets of data and features
- Much less prone to overfitting than a single Decision Tree
- `n_estimators` controls how many trees — more is better up to a point
- Works well out of the box with minimal tuning
- Can also report **feature importance** — which features matter most

---

## Requirements

```
pandas
scikit-learn
seaborn
matplotlib
```

Install with:
```bash
pip install pandas scikit-learn seaborn matplotlib
```

---

*This is part of the Machine Learning With Python From Scratch series. More topics coming soon!*
