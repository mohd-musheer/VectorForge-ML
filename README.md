# 🚀 VectorForgeML
### High-Performance Machine Learning Framework for R & C++

<p align="center">
  <img src="public/assets/images/VectorForgeML_Logo.png" alt="VectorForgeML Logo" width="200"/>
</p>

---

## 📌 Overview

**VectorForgeML** is a next-generation machine learning framework designed to bridge the gap between **R's simplicity** and **C++'s raw performance**. Built from scratch, it focuses on understanding the mathematical foundations of ML while delivering a scalable, production-ready systems architecture.

Unlike traditional R packages that simply wrap C libraries, **VectorForgeML** implements core algorithms using:
- **Raw C++ Pointers** for graph-based models (Decision Trees, Random Forests)
- **BLAS & LAPACK** for hardware-accelerated Linear Algebra
- **OpenMP** for multi-core parallelism
- **Zero-Copy** data exchange between R and C++

The goal is to provide a **Research-Ready Engine** that allows data scientists to build complex pipelines in milliseconds.

---

## ⚡ Key Features

- **🚀 Blazing Fast**: Optimized C++ backend ensures models train in record time compared to standard R implementations.
- **🔗 Modular Design**: Component-based architecture with full support for **Pipelines** (`ColumnTransformer`, `StandardScaler`, `PCA`, etc.).
- **🔧 Hardware Acceleration**: Native integration with BLAS/LAPACK for matrix operations.
- **📉 Memory Efficient**: Custom allocators and index-based views (no deep copies).
- **🛠 Zero Dependencies**: Core algorithms are implemented without heavy external ML dependencies.

---

## 📦 Installation

Install directly from GitHub using `remotes`:

```r
# Install remotes if not already installed
install.packages("remotes")

# Install VectorForgeML
remotes::install_github("mohd-musheer/VectorForgeML")

library(VectorForgeML)
```

---

## 🧠 Algorithms & Notebooks

We have implemented a wide range of algorithms, verified with real-world datasets on **Kaggle**.

### 🔹 Supervised Learning (Regression & Classification)

| Algorithm | Type | Description | Kaggle Demo |
|-----------|------|-------------|-------------|
| **Linear Regression** | Regression | OLS with BLAS/LAPACK optimization. | [View Notebook](https://www.kaggle.com/code/almusheer/linear-regression-vectorforgeml) |
| **Logistic Regression** | Classification | Gradient Descent with Sigmoid activation. | [View Notebook](https://www.kaggle.com/code/almusheer/logistic-regression-vectorforgeml) |
| **Ridge Regression** | Regression | L2 Regularized Linear Regression using Cholesky. | [View Notebook](https://www.kaggle.com/code/almusheer/ridge-regression-vectorforgeml) |
| **Softmax Regression** | Classification | Multi-class classification with Log-Sum-Exp trick. | [View Notebook](https://www.kaggle.com/code/almusheer/softmax-regression-vectorforgemlml) |
| **Decision Tree** | Reg/Class | Recursive partitioning with distinct C++ graph pointers. | [View Notebook](https://www.kaggle.com/code/almusheer/decision-tree-vectorforge-ml) |
| **Random Forest** | Ensemble | Parallelized ensemble of decision trees. | [View Notebook](https://www.kaggle.com/code/almusheer/randomforest-vectorforgeml) |
| **KNN** | Reg/Class | K-Nearest Neighbors with `std::partial_sort` optimization. | [View Notebook](https://www.kaggle.com/code/almusheer/knn-vectorforgeml) |

### 🔹 Unsupervised Learning

| Algorithm | Type | Description | Kaggle Demo |
|-----------|------|-------------|-------------|
| **K-Means Clustering** | Clustering | Lloyd's algorithm with efficient centroid updates. | [View Notebook](https://www.kaggle.com/code/almusheer/kmeans-vectorforgeml) |
| **PCA** | Dim. Reduction | Principal Component Analysis via SVD/Eigen Decomposition. | [View Notebook](https://www.kaggle.com/code/almusheer/decision-tree-vectorforge-ml) |

### 🔹 Utilities & Pipelines

| Feature | Description | Kaggle Demo |
|---------|-------------|-------------|
| **Pipeline** | Chain multiple steps (Preprocessing -> Model) into one object. | [View Notebook](https://www.kaggle.com/code/almusheer/pipeline-vectorforgeml) |
| **Preprocessing** | `StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `OneHotEncoder`. | Included in Pipeline Demo |
| **Metrics** | `accuracy_score`, `r2_score`, `f1_score`, `confusion_matrix`. | Included in all Demos |

---

## � Quick Start Example

Here is how you can build a powerful pipeline in just a few lines of R:

```r
library(VectorForgeML)

# 1. Load Data
data(iris)
X <- as.matrix(iris[, 1:4])
y <- as.numeric(iris[, 5]) - 1 # Convert to 0,1,2

# 2. Split Data
split <- train_test_split(X, y, test_size = 0.2)

# 3. Create a Pipeline
pipe <- Pipeline(steps = list(
   c("scaler", StandardScaler()),
   c("pca", PCA(n_components = 2)),
   c("model", SoftmaxRegression(epochs=1000, lr=0.1))
))

# 4. Train
pipe$fit(split$X_train, split$y_train)

# 5. Predict & Evaluate
preds <- pipe$predict(split$X_test)
acc <- accuracy_score(split$y_test, preds)
print(paste("Accuracy:", acc))
```

---

## 🏗 Project Structure

```bash
VectorForgeML/
├── R/                  # R interface & utility functions
├── src/                # High-performance C++ backend
│   ├── LinearRegression.cpp
│   ├── DecisionTree.cpp
│   ├── RandomForest.cpp
│   └── ...
├── include/            # C++ Header files
├── public/             # Documentation Website (VectorForgeML.com)
├── scripts/            # Build & Doc generation scripts
└── README.md           # Project Documentation
```

---

## 👨‍� Developer

**Mohd Musheer**  
*Lead Developer & Architect*

Passionate about High-Performance Computing (HPC) and System Design. VectorForgeML is a testament to building systems from first principles—combining rigorous mathematics with software engineering excellence.

---

## 📜 License

This project is licensed under the **MIT License** - free to use and modify.

---

<p align="center">
  Made with ❤️ by <b>Mohd Musheer</b>
</p>