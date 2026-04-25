# K Nearest Neighbors (KNN)

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is KNN?

K Nearest Neighbors is one of the simplest machine learning algorithms. To classify a new data point, it looks at the **K closest points** in the training data and takes a majority vote.

```
New point ★ — what class is it?

  o o         x x
    o  ★      x
  o o           x x

If K=3: find 3 nearest neighbors → majority wins
```

There is no "training" in the traditional sense — KNN memorizes all the training data and computes distances at prediction time.

### How to choose K?

- **Small K** (like 1 or 3): very sensitive to noise, can overfit
- **Large K**: smoother boundary but may underfit
- Usually try odd numbers to avoid ties (3, 5, 7, 11...)

---

## About This Notebook

Classifies **handwritten digits (0–9)** using the sklearn digits dataset. Each image is 8×8 = 64 pixel features.

### Dataset (`load_digits`)

| Property | Details |
|----------|---------|
| Samples | 1,797 images |
| Features | 64 pixel values |
| Classes | 10 (digits 0–9) |

---

## What the Code Does

### Step 1 — Load Dataset
```python
from sklearn.datasets import load_digits
digits = load_digits()
x = digits.data
y = digits.target
```

### Step 2 — Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

### Step 3 — Train KNN
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
```
`n_neighbors=11` means look at 11 nearest points and take the majority class.

### Step 4 — Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
y_predicted = knn.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
sn.heatmap(cm, annot=True)
```
Shows which digits the model confuses with each other (e.g. 3 vs 8, 4 vs 9).

### Step 5 — Classification Report
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predicted))
```
Shows precision, recall, and F1-score for each digit class — more detailed than just accuracy.

---

## Key Takeaways

- KNN is simple — no training, just memorise and compare at prediction time
- `n_neighbors` is the most important parameter to tune
- Use odd K values to avoid ties
- KNN can be slow on large datasets since it computes distances to every training point
- Works well for image recognition where similar images have similar pixel values

---

## Requirements

```
scikit-learn
seaborn
matplotlib
```

Install with:
```bash
pip install scikit-learn seaborn matplotlib
```

---

*This is part of the Machine Learning With Python From Scratch series. More topics coming soon!*
