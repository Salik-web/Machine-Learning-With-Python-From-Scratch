# K-Means Clustering

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is K-Means Clustering?

K-Means is an **unsupervised learning** algorithm — there are no labels. It finds natural groupings in your data by partitioning it into K clusters.

### How it works

1. Pick K random points as initial cluster centres
2. Assign every data point to the nearest centre
3. Recalculate each centre as the mean of all points assigned to it
4. Repeat steps 2–3 until centres stop moving

```
Before clustering:       After K-Means (K=3):
  . . . . . .              o o o x x .
  . . . . . .              o o o x x .
  . . . . . .              o o x x . .
```

### The Elbow Method

How do you choose the right K? Plot the **Sum of Squared Errors (SSE)** for different values of K. The SSE drops sharply and then levels off — the "elbow" point is the best K.

---

## About This Notebook

Groups **Iris flowers** into clusters using only sepal measurements — without using the actual labels.

### Dataset (`load_iris`)

| Property | Details |
|----------|---------|
| Samples | 150 flowers |
| Features used | sepal length, sepal width (2 of 4) |
| Classes | 3 (but we pretend not to know this) |

---

## What the Code Does

### Step 1 — Load and Prepare Data
```python
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df = df.drop(['petal length (cm)', 'petal width (cm)'], axis='columns')
```
Only sepal features are used to keep the visualisation 2D.

### Step 2 — Scatter Plot (Before Clustering)
```python
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'])
```
Shows the raw data with no labels.

### Step 3 — Apply KMeans
```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['sepal length (cm)', 'sepal width (cm)']])
df['Cluster'] = y_predicted
```
`fit_predict` trains the model and returns a cluster label (0, 1, or 2) for each row.

### Step 4 — Check Cluster Centres
```python
km.cluster_centers_
```
The coordinates of the 3 final cluster centres.

### Step 5 — MinMaxScaler (Normalisation)
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['sepal length (cm)'] = scaler.fit_transform(df[['sepal length (cm)']])
```
Features on different scales can bias the distance calculation. MinMaxScaler brings all values to the range 0–1 so no single feature dominates.

### Step 6 — Re-run KMeans After Scaling
```python
km1 = KMeans(n_clusters=3)
y_predicted1 = km1.fit_predict(df[['sepal length (cm)', 'sepal width (cm)']])
```
Scaling often improves clustering results.

### Step 7 — Elbow Method
```python
sse = []
for k in range(1, 10):
    km = KMeans(n_clusters=k)
    km.fit(df[['sepal length (cm)', 'sepal width (cm)']])
    sse.append(km.inertia_)

plt.plot(range(1, 10), sse)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
```
`inertia_` is the SSE. The elbow in the plot shows the optimal K.

---

## Key Takeaways

- K-Means is unsupervised — no labels needed, it finds patterns on its own
- You must choose K in advance — use the Elbow Method to find the best value
- **Always scale your features** before clustering — different scales distort distances
- `MinMaxScaler` brings features to 0–1; `StandardScaler` brings to mean=0, std=1
- K-Means works best when clusters are roughly circular and similar in size

---

## Requirements

```
pandas
scikit-learn
matplotlib
```

Install with:
```bash
pip install pandas scikit-learn matplotlib
```

---

*This is part of the Machine Learning With Python From Scratch series. More topics coming soon!*
