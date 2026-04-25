# Grid Search CV

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is Grid Search CV?

When training a model, you have parameters like `C`, `kernel`, `max_depth` to choose. Picking these manually by trial and error is slow and unreliable.

**Grid Search CV** automates this — you give it a grid of all parameter combinations you want to try, and it trains and evaluates every combination using cross-validation, then tells you which one performed best.

```
Parameter grid:
  C:      [1, 10, 20, 50]
  kernel: ['linear', 'rbf']

Grid Search tries all 8 combinations:
  (C=1,  kernel=linear) → CV score
  (C=1,  kernel=rbf)    → CV score
  (C=10, kernel=linear) → CV score
  ... and so on

Best params: C=10, kernel=rbf → 98%
```

### RandomizedSearchCV

Instead of trying every combination, **RandomizedSearchCV** randomly samples `n_iter` combinations. Faster when the parameter space is large, and often finds a result nearly as good as full Grid Search.

---

## About This Notebook

Uses the **digits dataset** (handwritten digit recognition, 64 features) to demonstrate 3 approaches to parameter tuning, ending with comparing multiple model types using Grid Search.

---

## What the Code Does

### Approach 1 — Manual Trial and Error
```python
svm = SVC(kernel='rbf', C=10, gamma='auto')
svm.fit(X_train, y_train)
svm.score(X_test, y_test)

svm1 = SVC(kernel='linear', C=10, gamma='auto')
```
Try one combination at a time — slow and subjective.

### Approach 2 — For Loop
```python
kernel = ['rbf', 'linear']
c = [1, 10, 20, 50]
avg_score = {}

for kval in kernel:
    for cval in c:
        cv_score = cross_val_score(SVC(kernel=kval, C=cval, gamma='auto'), digits.data, digits.target, cv=5)
        avg_score[kval + '_' + str(cval)] = np.average(cv_score)
```
Better — tries all combinations and stores results in a dictionary.

### Approach 3 — GridSearchCV
```python
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(SVC(gamma='auto'), {
    'C': [1, 10, 20, 50],
    'kernel': ['linear', 'rbf']
}, cv=5, return_train_score=False)
clf.fit(digits.data, digits.target)
```
The cleanest approach — does everything automatically.

### View Results
```python
df2 = pd.DataFrame(clf.cv_results_)
df2[['param_C', 'param_kernel', 'mean_test_score']]
clf.best_params_
clf.best_score_
```

### RandomizedSearchCV
```python
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(SVC(gamma='auto'), {
    'C': [1, 10, 20, 50],
    'kernel': ['linear', 'rbf']
}, cv=5, n_iter=3)
rs.fit(digits.data, digits.target)
rs.best_params_
```
Only tries 3 random combinations instead of all 8.

### Compare Multiple Model Types
```python
model_params = {
    'svm': {'model': SVC(gamma='auto'), 'params': {'C': [1,10,20,30], 'kernel': ['linear','rbf']}},
    'logistic_reg': {'model': LogisticRegression(), 'params': {'C': [10,1,20,30]}},
    'Decision_tree': {'model': DecisionTreeClassifier(), 'params': {'max_depth': [5,10,20,30]}},
    'Random_forest': {'model': RandomForestClassifier(), 'params': {'max_depth': [5,10,20,30]}}
}

scores = []
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5)
    clf.fit(digits.data, digits.target)
    scores.append({'model': model_name, 'best_score': clf.best_score_, 'best_params': clf.best_params_})

pd.DataFrame(scores)
```
Runs Grid Search on 4 different models and compares the best score from each.

---

## Key Takeaways

- Never tune parameters manually — use Grid Search CV
- `best_params_` gives you the winning combination; `best_score_` gives the CV score
- `RandomizedSearchCV` is faster for large grids at the cost of possibly missing the best combination
- `cv_results_` as a DataFrame lets you see scores for every combination tried
- You can run Grid Search across completely different model types to find the best algorithm

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
