# Naive Bayes — Wine Classification

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is Naive Bayes?

Naive Bayes is a classification algorithm based on **Bayes' Theorem**. It calculates the probability that a sample belongs to each class, then picks the class with the highest probability.

It is called "naive" because it assumes all features are **independent** — knowing one feature tells you nothing about another. This is rarely true in real data, but the algorithm still works very well in practice.

### GaussianNB vs MultinomialNB

| | GaussianNB | MultinomialNB |
|--|------------|--------------|
| **Use for** | Continuous numeric data | Discrete counts (like word frequencies) |
| **Assumption** | Features follow a normal distribution | Features are non-negative counts |

---

## About This Notebook

Classifies **wine into 3 types** based on 13 chemical properties using the built-in sklearn Wine dataset.

### Dataset (`load_wine`)

| Property | Details |
|----------|---------|
| Samples | 178 wines |
| Features | 13 (alcohol, malic acid, ash, etc.) |
| Classes | 3 wine types |
| Source | Built into sklearn |

---

## What the Code Does

### Step 1 — Load Dataset
```python
from sklearn.datasets import load_wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
```

### Step 2 — Add Target and Names
```python
df['Target'] = wine.target
df['wine_names'] = [wine.target_names[i] for i in wine.target]
```
Adds both the numeric target and the readable wine class name.

### Step 3 — Prepare Features and Target
```python
df = df.drop(['Target', 'wine_names'], axis='columns')
x = df
y = wine.target
```

### Step 4 — Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

### Step 5 — Train GaussianNB
```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
model.score(X_test, y_test)
```
GaussianNB assumes the features follow a normal (bell-curve) distribution — suited for continuous measurements like chemical properties.

### Step 6 — Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
```
Shows exactly which wine types the model confused.

### Step 7 — Try MultinomialNB
```python
from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X_train, y_train)
NB.score(X_test, y_test)
```
MultinomialNB is designed for count data — it will perform worse here since wine measurements are continuous, not counts.

---

## Key Takeaways

- **GaussianNB** is for continuous numeric data — use it for measurements
- **MultinomialNB** is for count data — use it for text/word frequencies
- Naive Bayes is fast, simple, and works well even with small datasets
- The confusion matrix shows which classes the model struggles to separate

---

## Requirements

```
pandas
numpy
scikit-learn
seaborn
matplotlib
```

Install with:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

---

*This is part of the Machine Learning With Python From Scratch series. More topics coming soon!*
