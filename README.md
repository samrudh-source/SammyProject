# Academic Performance Principal Pattern Analyzer

## Aim

This B.Tech Linear Algebra & Vector Calculus mini-project is a Python/Streamlit application that analyzes multidimensional student academic data and identifies dominant directions of variation using Principal Component Analysis (PCA).

## Problem statement and objectives

Academic records contain correlated measurements such as Mathematics, Physics, Programming, and Attendance. The project represents these observations as vectors, computes their covariance structure, derives orthogonal eigen-directions, and projects students into a lower-dimensional representation. It aims to:

1. Make every PCA stage visible and mathematically traceable.
2. Demonstrate matrices, vector spaces, means, centering, covariance, eigenvalues, eigenvectors, orthogonality, projection, principal components, and dimensionality reduction.
3. Provide factual visual and numerical pattern interpretation without unsupported student judgments.
4. Verify the manual calculation against scikit-learn.

## Mathematical formulation

For `n` students and `m` numerical features, the student-feature matrix is `X ∈ R^(n×m)`.

* Mean: `μ_j = (1/n) Σ X_ij`
* Centering: `X_c = X − μ`
* Covariance: `C = (1/(n−1)) X_cᵀ X_c`
* Eigenvalue equation: `C v = λ v`
* Orthogonality: `v_iᵀ v_j = 0`, for `i ≠ j`
* Projection: `Z = X_c V_k`
* Explained variance ratio: `EVR_i = λ_i / Σ λ_j`
* Cumulative explained variance: `CEV_k = Σ(i=1 to k) EVR_i`

The application executes this chain explicitly:

**DATA → MATRIX → CENTERING → COVARIANCE MATRIX → EIGENVALUES + EIGENVECTORS → ORTHOGONAL PRINCIPAL DIRECTIONS → PROJECTION → REDUCED REPRESENTATION → INTERPRETATION**

## Technology

Python 3, NumPy, Pandas, Matplotlib, Streamlit, and scikit-learn (verification only). Tests use pytest.

## Project structure

```text
academic-pca-analyzer/
├── app.py
├── requirements.txt
├── README.md
├── data/sample_students.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── pca_calculator.py
│   ├── visualization.py
│   └── verification.py
└── tests/test_pca.py
```

## Installation and running

In VS Code, open the project folder, create/activate a virtual environment, and run:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
streamlit run app.py
```

Then open the local URL displayed by Streamlit. The sample dataset loads automatically. Use **Upload CSV** or **Load Sample Dataset**, then **Run PCA**.

## Sample input format

CSV files need at least two numerical academic columns. Optional `Student Name` and `Roll Number` columns are treated as labels and excluded from PCA:

```csv
Student Name,Roll Number,Mathematics,Physics,Programming,Attendance
Aarav Sharma,101,88,84,92,96
Diya Patel,102,76,79,85,91
```

Missing numerical values are filled with that feature's mean. Other non-numerical columns are reported and ignored.

## Expected output

The dashboard has ten tabs: Home, Dataset, Centering, Covariance Matrix, Eigen Analysis, Principal Components, Projection & Reduction, Visualizations, Verification, and Mathematical Explanation. It displays the original and centered matrices, means, covariance matrix, sorted eigenvalues/eigenvectors, explained variance, `VᵀV`, loading tables, projected scores, retained variance, scree/variance/scatter/loading plots, and a verification table.

## File responsibilities

* `app.py`: Streamlit layout, controls, formulas, tables, plots, and interpretations.
* `data_loader.py`: CSV/sample loading, identifier exclusion, validation, and missing-value handling.
* `preprocessing.py`: feature means and explicit matrix centering.
* `pca_calculator.py`: covariance, `np.linalg.eigh`, descending sorting, orthogonality, and projection.
* `visualization.py`: required Matplotlib figures.
* `verification.py`: independent scikit-learn comparison with eigenvector sign alignment.
* `tests/test_pca.py`: tests for sample data, centering, orthogonality, projection, missing values, and numerical agreement.

## Verification methodology

The manual covariance matrix uses `(X_c.T @ X_c)/(n-1)`, then `np.linalg.eigh(C)` is sorted in descending order. scikit-learn PCA is fitted separately only for comparison. Explained variances and ratios should match to floating-point precision. Scores are sign-aligned because both `v` and `−v` are valid representations of an eigenvector.

## Results and interpretation

PC1 is the direction with the largest eigenvalue and therefore the largest share of observed variation. Feature loading magnitudes identify which academic coordinates contribute most to each component. Points close together in the PC1–PC2 plot have similar retained coordinates. These are descriptive numerical patterns, not claims that a student is weak, will fail, or will achieve a particular outcome.

## Limitations and future enhancements

PCA is sensitive to scale and outliers; this version uses raw numeric values and does not make predictive claims. The covariance approach assumes at least two observations and two features. Future work could add optional standardization, robust outlier diagnostics, downloadable reports, larger datasets, and user-selectable labeling.

## Viva questions and answers

**Q: Why center the data?** A: Centering makes the origin the feature mean, so covariance measures variation around the average rather than around zero.

**Q: Why use `eigh` instead of `eig`?** A: A covariance matrix is symmetric, and `eigh` is specialized for symmetric/Hermitian matrices and returns real eigenvalues.

**Q: What does an eigenvalue mean?** A: It is the variance captured along its corresponding eigenvector.

**Q: Why sort eigenvalues?** A: Sorting descending ranks directions from greatest to least variation, so the first components are the most informative.

**Q: Why are principal directions orthogonal?** A: A symmetric covariance matrix has orthogonal eigenvectors; this gives uncorrelated coordinate directions.

**Q: What is a loading?** A: An eigenvector coefficient showing how strongly an original feature contributes to a principal direction.

**Q: What is dimensionality reduction?** A: Replacing the original `m` coordinates with the first `k` principal coordinates while reporting the variance retained.

**Q: Why can manual and sklearn scores have opposite signs?** A: Eigenvectors are sign-ambiguous: `v` and `−v` define the same axis.

**Q: Is PCA a prediction model?** A: No. It is an unsupervised descriptive transformation that summarizes variation.

# Academic Performance Principal Pattern Analyzer

A Python and Streamlit based PCA application for analyzing multidimensional academic data.

## 🚀 Live Demo

[Open the Streamlit Application](https://sammyproject-v3jjnlynyizv2hulyhquky.streamlit.app/)

## 🖥️ Application Preview

![Application Preview](APLLICATION OVERVIEW.png)

## 🧮 Mathematical Pipeline

DATA
↓
MATRIX
↓
CENTERING
↓
COVARIANCE MATRIX
↓
EIGENVALUES & EIGENVECTORS
↓
PRINCIPAL COMPONENTS
↓
PROJECTION
↓
REDUCED REPRESENTATION

## 💻 Source Code

https://github.com/samrudh-source/SammyProject
