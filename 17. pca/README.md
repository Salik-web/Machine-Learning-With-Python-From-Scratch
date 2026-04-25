# PCA — Principal Component Analysis

> Part of the **Machine Learning With Python From Scratch** series — a beginner-friendly guide to machine learning concepts with hands-on Python code.

---

## What is PCA?

**Principal Component Analysis (PCA)** is a dimensionality reduction technique. When you have many features, some may be redundant or correlated. PCA transforms your features into a smaller set of new features called **principal components** that still capture most of the information.

### Why reduce dimensions?

- Fewer features = faster training
- Removes noise and redundant information
- Easier to visualise (2D or 3D)
- Can improve model accuracy by removing irrelevant features

### How it works

PCA finds directions (components) in the data where variance is highest — the directions that capture the most information. It ranks them by how much variance they explain and lets you keep only the top N.

```
11 original features
         ↓ PCA
    7 components that explain 95% of variance
```

### Explained Variance Ratio

Each component explains some percentage of the total variance. `PCA(0.95)` automatically selects enough components to explain 95% of the variance in the data.

---

## About This Notebook

Uses **heart disease data** to show how PCA can reduce 11 features to fewer components while maintaining model accuracy.

### Dataset (`pca_ex.csv`)

| Column | Description |
|--------|-------------|
| `Age` | Patient age |
| `Sex` | Gender |
| `ChestPainType` | Type of chest pain |
| `RestingBP` | Resting blood pressure |
| `Cholesterol` | Serum cholesterol |
| `FastingBS` | Fasting blood sugar |
| `RestingECG` | Resting ECG results |
| `MaxHR` | Maximum heart rate |
| `ExerciseAngina` | Exercise-induced angina |
| `Oldpeak` | ST depression |
| `ST_Slope` | ST segment slope |
| `HeartDisease` | **Target** — 1=disease, 0=no disease |

---

## What the Code Does

### Step 1 — Load and Remove Outliers
```python
df_numeric = df.select_dtypes(include=[np.number])
df_clean = df[(np.abs(stats.zscore(df_numeric)) < 3).all(axis=1)]
```

### Step 2 — Encode Categorical Columns
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_clean = df_clean.copy()
col_to_numeric = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df_clean[col_to_numeric] = df_clean[col_to_numeric].apply(le.fit_transform)
```

### Step 3 — Prepare Features and Target
```python
x = df_clean.drop('HeartDisease', axis='columns')
y = df_clean['HeartDisease']
```

### Step 4 — StandardScaler (Required Before PCA)
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
```
PCA is sensitive to scale — always standardize first so no feature dominates.

### Step 5 — Train Baseline Model (Without PCA)
```python
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)
```
Records accuracy before PCA as a reference point.

### Step 6 — Apply PCA (95% Variance)
```python
from sklearn.decomposition import PCA
pca = PCA(0.95)
X_pca = pca.fit_transform(x_scaled)
pca.n_components_          # how many components were selected
pca.explained_variance_ratio_  # how much each component explains
```
`PCA(0.95)` automatically picks the fewest components needed to retain 95% of information.

### Step 7 — Train Model With PCA Features
```python
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=30)
log_reg2 = LogisticRegression(C=10)
log_reg2.fit(X_train_pca, y_train)
log_reg2.score(X_test_pca, y_test)
```
Compare accuracy — PCA version should be close to or better than the baseline.

### Step 8 — PCA with Fixed Number of Components
```python
pca2 = PCA(n_components=5)
X_pca2 = pca2.fit_transform(x_scaled)
pca2.explained_variance_ratio_
```
Explicitly choose 5 components and see how much variance they explain.

---

## Key Takeaways

- PCA reduces dimensions while preserving as much information as possible
- **Always scale data before PCA** — PCA is based on variance, and scale affects variance
- `PCA(0.95)` automatically selects components for 95% variance retention
- `pca.n_components_` tells you how many components were selected
- `pca.explained_variance_ratio_` shows the contribution of each component
- PCA trades interpretability for performance — you lose the original feature names

---

## Requirements

```
pandas
numpy
scipy
scikit-learn
matplotlib
```

Install with:
```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

---

*This is part of the Machine Learning With Python From Scratch series. More topics coming soon!*
