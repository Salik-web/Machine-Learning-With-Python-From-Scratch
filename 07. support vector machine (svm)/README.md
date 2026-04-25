# Support Vector Machine (SVM)

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is SVM?

A **Support Vector Machine** is a classification algorithm that finds the best boundary (called a **hyperplane**) to separate two or more classes of data.

The key idea is to find the boundary that has the **maximum margin** — the largest gap between the nearest data points of each class. Those nearest points are called **support vectors**.

```
Class A:  o  o  o
                   |  ← hyperplane (decision boundary)
Class B:  x  x  x
```

SVM tries to maximize the distance between the support vectors and the hyperplane — this makes it more robust to new data.

### Why SVM for image recognition?

Images are high-dimensional data (64 pixel values per image in this example). SVM handles high-dimensional data well and is effective even when there are more features than samples.

---

## About This Notebook

In this notebook we classify **handwritten digits (0–9)** using the built-in sklearn digits dataset. Each image is 8×8 pixels = 64 features.

### Dataset (`load_digits`)

| Property | Details |
|----------|---------|
| Samples | 1,797 images |
| Features | 64 (8×8 pixel values) |
| Classes | 10 (digits 0–9) |
| Source | Built into sklearn |

---

## What the Code Does

### Step 1 — Load and Visualize the Dataset
```python
digits = load_digits()

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])
```
`digits.images` contains the raw 8×8 pixel grids. `plt.matshow` displays each one as a grayscale image so you can see what the data looks like.

### Step 2 — Inspect the Data
```python
digits.target_names   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
digits.data[0]        # 64 pixel values for the first image
```
`digits.data` is a flat array of 64 values per image. `digits.target_names` shows the 10 class labels.

### Step 3 — Create a DataFrame
```python
df = pd.DataFrame(digits.data, columns=range(64))
```
Organizes the pixel data into a readable table with 64 columns (one per pixel).

### Step 4 — Prepare Target
```python
y = digits.target
```
The label for each image — which digit (0–9) it actually is.

### Step 5 — Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
```
80% for training, 20% for testing.

### Step 6 — Train the Default SVM Model
```python
model = SVC()
model.fit(X_train, y_train)
model.score(X_test, y_test)
```
`SVC()` with default settings — uses RBF kernel, C=1, gamma='scale'.

### Step 7 — Tune C (Regularization)
```python
model_c = SVC(C=4)
model_c.fit(X_train, y_train)
model_c.score(X_test, y_test)
```
`C` controls how strictly the model tries to classify every training point correctly:
- **High C** → fits training data tightly, risk of overfitting
- **Low C** → allows some misclassification, more generalized

### Step 8 — Tune Gamma
```python
model_g = SVC(gamma=1)
model_g.fit(X_train, y_train)
model_g.score(X_test, y_test)
```
`gamma` controls how far the influence of a single training point reaches:
- **High gamma** → each point influences only nearby points → complex boundary → overfitting
- **Low gamma** → each point influences a wider area → smoother boundary

### Step 9 — Try RBF Kernel
```python
model_rbf = SVC(kernel='rbf')
model_rbf.fit(X_train, y_train)
model_rbf.score(X_test, y_test)
```
The **kernel** defines how the SVM maps data into higher dimensions to find the separating boundary:
- `rbf` (Radial Basis Function) — works well for non-linear data, the most common choice
- `linear` — works when data is linearly separable
- `poly` — polynomial boundary

---

## Key Parameters Summary

| Parameter | What it controls | Too high | Too low |
|-----------|-----------------|----------|---------|
| `C` | Regularization strength | Overfitting | Underfitting |
| `gamma` | Influence radius per point | Overfitting | Underfitting |
| `kernel` | Type of decision boundary | — | — |

---

## Key Takeaways

- SVM finds the boundary with the **maximum margin** between classes
- Works well with **high-dimensional data** like images
- The **kernel** transforms data so a linear boundary can separate non-linear classes
- `C` and `gamma` are the most important parameters to tune
- Use **Grid Search CV** to systematically find the best combination of parameters

---

## Requirements

```
pandas
matplotlib
scikit-learn
```

Install with:
```bash
pip install pandas matplotlib scikit-learn
```

---

*This is part of the Machine Learning With Python From Scratch series. More topics coming soon!*
