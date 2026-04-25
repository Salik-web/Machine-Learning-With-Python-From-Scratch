# Naive Bayes — Email Spam Detection

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is Naive Bayes?

Naive Bayes is a classification algorithm based on **Bayes' Theorem** — it calculates the probability that a piece of data belongs to a class given the features it contains.

The "naive" part means it assumes all features are **independent** of each other. For text classification this is technically not true (words are related), but in practice it works extremely well.

### Why is it great for text/spam detection?

- Very fast to train even on large datasets
- Works well with high-dimensional data (thousands of words as features)
- Performs surprisingly well despite the naive independence assumption

### MultinomialNB vs GaussianNB

| | MultinomialNB | GaussianNB |
|--|--------------|------------|
| **Use for** | Text (word counts) | Continuous numeric data |
| **Input** | Integer counts | Any real numbers |

---

## About This Notebook

Classifies emails as **spam or not spam** using real email message data.

### Dataset (`naive_bayes_email.csv`)

| Column | Description |
|--------|-------------|
| `Category` | `spam` or `ham` (not spam) |
| `Message` | The raw email text |

---

## What the Code Does

### Step 1 — Load Data
```python
df = pd.read_csv('naive_bayes_email.csv')
df.groupby('Category').describe()
```
Explore how many spam vs ham emails exist.

### Step 2 — Create Numeric Target
```python
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
```
Converts `spam`/`ham` text to `1`/`0` for the model.

### Step 3 — Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['spam'], test_size=0.25)
```
Splits the raw email text into training and test sets.

### Step 4 — CountVectorizer
```python
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
```
Models can't read text — CountVectorizer converts each email into a vector of word counts. Each unique word becomes a feature (column), and the value is how many times that word appears.

### Step 5 — Train MultinomialNB
```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count, y_train)
```
MultinomialNB is designed for count data — perfect for word frequency vectors.

### Step 6 — Predict Custom Emails
```python
emails = [
    'Hey mohan, can we get together to watch football game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you!'
]
```
Tests the model on hand-written examples to see if it catches spam.

### Step 7 — Score on Test Set
```python
X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)
```
Note: use `v.transform()` (not `fit_transform`) on test data — the vocabulary was already learned from training data.

### Step 8 — sklearn Pipeline (Cleaner Approach)
```python
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorize', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)
clf.predict(emails)
clf.score(X_test, y_test)
```
Pipeline chains CountVectorizer and the model together — you can pass raw text directly without manually calling transform. Cleaner and production-ready.

---

## Key Takeaways

- Naive Bayes is one of the best algorithms for text classification
- **CountVectorizer** converts text to word count vectors for the model
- Use `fit_transform` on training data, `transform` only on test data
- **Pipeline** makes it cleaner — handles vectorization and prediction in one step
- MultinomialNB works with word counts; GaussianNB works with continuous numbers

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
