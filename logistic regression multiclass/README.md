# Logistic Regression — Multiclass Classification

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is Multiclass Classification?

In binary classification, the model predicts between **two classes** — Yes or No, 0 or 1.

In **Multiclass Classification**, the model predicts between **three or more classes**. For example:
- Is this flower a Setosa, Versicolor, or Virginica?
- Is this digit a 0, 1, 2 ... or 9?
- Is this news about Sports, Politics, or Technology?

Logistic Regression handles multiclass problems using a strategy called **One vs Rest (OvR)** — it trains one binary classifier per class and picks the one with the highest confidence.

---

## What is the Iris Dataset?

The Iris dataset is one of the most famous datasets in machine learning. It contains measurements of **150 flowers** from 3 different species.

| Feature | Description |
|---------|-------------|
| `sepal length (cm)` | Length of the sepal |
| `sepal width (cm)` | Width of the sepal |
| `petal length (cm)` | Length of the petal |
| `petal width (cm)` | Width of the petal |
| `target` | Flower species (0, 1, or 2) |

The 3 target classes:
- `0` = Setosa
- `1` = Versicolor
- `2` = Virginica

50 flowers per class, 150 total. No missing values — clean and ready to use.

---

## What the Code Does

### Step 1 — Import Libraries and Load Dataset
```python
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.feature_names)   # column names
print(iris.target_names)    # class names
```
The Iris dataset is built into sklearn — no CSV needed.

### Step 2 — Create a DataFrame
```python
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
```
Converts the dataset into a readable pandas DataFrame.

### Step 3 — Add Flower Names for Readability
```python
df['flower_name'] = [iris.target_names[i] for i in iris.target]
```
Adds a human-readable column showing the actual flower name instead of just 0, 1, 2.

### Step 4 — Explore the Data
```python
df[50:100].head()
```
Rows 50–100 are all Versicolor — useful to visually inspect one class at a time.

### Step 5 — Prepare Features and Target
```python
df = df.drop(['flower_name'], axis='columns')

a = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df['target']
```
`a` = all 4 measurements (input features)
`y` = flower species (target)

### Step 6 — Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(a, y, test_size=0.2, random_state=42)
```
- 80% training, 20% testing
- `random_state=42` ensures the same split every time you run

### Step 7 — Train the Model
```python
model = LogisticRegression()
model.fit(X_train, y_train)
```
Logistic Regression automatically detects there are 3 classes and uses One vs Rest strategy internally.

### Step 8 — Evaluate Accuracy
```python
model.score(X_test, y_test)
```
Returns the accuracy score on unseen test data.

### Step 9 — Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
```
The confusion matrix shows exactly where the model is right and where it makes mistakes.

### Step 10 — Visualize with Heatmap
```python
import seaborn as sn
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
```
A heatmap makes the confusion matrix easy to read:
- **Diagonal** = correct predictions
- **Off-diagonal** = wrong predictions (model confused one class for another)

---

## Understanding the Confusion Matrix

For a 3-class problem it looks like this:

```
              Predicted
              Setosa  Versicolor  Virginica
Actual Setosa  [ 10       0           0   ]
  Versicolor   [  0       9           1   ]
   Virginica   [  0       0          10   ]
```

- All 10 Setosas predicted correctly ✅
- 9 Versicolors correct, 1 wrongly predicted as Virginica ❌
- All 10 Virginicas correct ✅

Versicolor and Virginica are the hardest to separate because their measurements overlap in the real world.

---

## Key Takeaways

- Logistic Regression works for multiclass problems — not just binary
- The Iris dataset is a great starting point for classification
- **Confusion matrix** is more informative than accuracy alone — it shows which classes are being confused
- `random_state` in train_test_split ensures reproducible results
- Seaborn heatmap makes the confusion matrix much easier to interpret visually

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

*This is part of the Machine Learning With Python From Scratch series. More topics coming soon!*
