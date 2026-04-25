# K-Fold Cross Validation

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is K-Fold Cross Validation?

When you do a simple train/test split, your score depends heavily on **which rows randomly ended up in the test set**. Run it again with a different `random_state` and you might get a completely different accuracy.

**K-Fold Cross Validation** solves this by splitting the data into K equal parts (folds), training K times — each time using a different fold as the test set — and averaging the scores.

```
Data: [fold1] [fold2] [fold3]

Run 1: Train on fold2+fold3, test on fold1 → score1
Run 2: Train on fold1+fold3, test on fold2 → score2
Run 3: Train on fold1+fold2, test on fold3 → score3

Final score = average(score1, score2, score3)
```

This gives a much more reliable estimate of how well your model generalises.

### StratifiedKFold

Regular KFold splits randomly. **StratifiedKFold** ensures each fold has the same class distribution as the full dataset — important for imbalanced datasets.

---

## About This Notebook

Uses the built-in **Iris dataset** (150 flowers, 3 species) to compare 4 models using K-Fold Cross Validation.

---

## What the Code Does

### Step 1 — Load Dataset and Basic Train/Test Split
```python
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
```
First shows the traditional approach — one split, one score.

### Step 2 — Train 4 Models with Simple Split
```python
lr = LogisticRegression()
svm = SVC(gamma=5)
decision = DecisionTreeClassifier()
forest = RandomForestClassifier(n_estimators=60)
```
Each model is trained and scored once. These scores vary each run.

### Step 3 — Understand KFold
```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
for train_index, test_index in kf.split([1,4,6,3,7,4,5,9,8]):
    print(train_index, test_index)
```
Shows how KFold splits indices into 3 groups.

### Step 4 — Manual Cross Validation with get_score
```python
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)
```
A helper function to train and score any model in one line.

### Step 5 — StratifiedKFold Manual Loop
```python
fold = StratifiedKFold(n_splits=3)
```
Loops through each fold manually, training and scoring each model.

### Step 6 — cross_val_score (Shortcut)
```python
from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(), iris.data, iris.target, cv=3)
```
`cross_val_score` does everything automatically — splits, trains, scores, and returns an array of scores for each fold.

### Step 7 — Compare All Models
```python
score1 = cross_val_score(LogisticRegression(), iris.data, iris.target, cv=3)
score2 = cross_val_score(SVC(), iris.data, iris.target, cv=3)
score3 = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target, cv=3)
np.average(score2)
```
Average the scores across all folds for a reliable comparison.

---

## Key Takeaways

- Simple train/test split gives a score that varies with `random_state` — unreliable
- K-Fold uses all data for both training and testing across K runs
- `cross_val_score` is the easiest way to do K-Fold — one line
- `StratifiedKFold` is better for classification to keep class balance in each fold
- Always use `np.average()` on the scores array to get the final number

---

## Requirements

```
scikit-learn
numpy
```

Install with:
```bash
pip install scikit-learn numpy
```

---

*This is part of the Machine Learning With Python From Scratch series. More topics coming soon!*
