# Instructions: Academic Performance Principal Pattern Analyzer

## 1. Prerequisites

Install the following before running the project:

- Python 3.10 or newer
- Visual Studio Code
- Internet access for installing Python packages

Check Python:

```bash
python --version
```

On some systems, use:

```bash
python3 --version
```

## 2. Open the project in VS Code

1. Open Visual Studio Code.
2. Select **File → Open Folder**.
3. Select the project folder containing `app.py`.
4. Open the VS Code integrated terminal using **Terminal → New Terminal**.

The terminal should be located at the project root:

```text
SammyProject/
```

You can verify the location with:

```bash
pwd
```

On Windows PowerShell, use:

```powershell
Get-Location
```

## 3. Create a virtual environment

Create an isolated Python environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

After activation, the terminal usually shows `(.venv)` at the beginning of the prompt.

## 4. Install required packages

Install all project dependencies from `requirements.txt`:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project uses:

- NumPy for matrix and vector calculations
- Pandas for CSV and table handling
- Matplotlib for charts
- Streamlit for the web dashboard
- Scikit-learn only for PCA verification
- Pytest for automated tests

## 5. Run the automated tests

From the project root, run:

```bash
python -m pytest
```

A successful run should report all tests passing. The tests verify:

- Sample data loading
- Exclusion of student names and roll numbers from PCA features
- Mean centering
- Manual PCA eigenvalues and explained variance
- Eigenvector orthogonality
- Projection dimensions
- Missing-value imputation
- Agreement with scikit-learn PCA

## 6. Start the Streamlit application

Run:

```bash
streamlit run app.py
```

Streamlit displays a local URL similar to:

```text
Local URL: http://localhost:8501
```

Open that URL in a web browser. To stop the application, return to the terminal and press:

```text
Ctrl+C
```

## 7. Use the dashboard

### Step 1: Select data

The application loads `data/sample_students.csv` automatically.

You may also:

- Click **Load Sample Dataset** in the sidebar.
- Click **Upload CSV** and select your own CSV file.
- Click **Reset** to restore the bundled sample dataset.

### Step 2: Validate the dataset

The application checks:

- Whether the dataset is empty
- Whether at least two numerical feature columns exist
- Whether numerical values are missing
- Whether non-numerical columns are present

`Student Name`, `Roll Number`, `Name`, `ID`, and similar identifier columns are not used as PCA features.

Missing numerical values are replaced by the mean of their corresponding feature.

### Step 3: Run PCA

Click **Run PCA** in the sidebar.

The application then performs the following sequential process:

```text
Student Data
↓
Student-Feature Matrix X
↓
Centered Matrix Xc
↓
Covariance Matrix C
↓
Eigenvalues and Eigenvectors
↓
Principal Directions
↓
Projection Scores Z
↓
Reduced Representation
```

## 8. Dashboard tabs

### Home

Introduces the project and displays the complete PCA workflow.

### Dataset

Displays:

- Number of students
- Number of PCA features
- Student-feature matrix
- Numerical academic columns used for analysis

### Centering

Displays:

- Feature means
- Original matrix
- Centered matrix

The formula shown is:

```text
μj = (1/n) Σ Xij
Xc = X − μ
```

### Covariance Matrix

Displays the manually calculated covariance matrix:

```text
C = (1/(n−1)) XcᵀXc
```

It also displays a covariance heatmap.

### Eigen Analysis

Displays:

- Sorted eigenvalues
- Eigenvectors
- Individual explained variance
- Cumulative explained variance

The eigenvalue relationship is:

```text
C v = λ v
```

### Principal Components

Displays:

- `VᵀV` orthogonality matrix
- A slider for selecting the number of principal components
- Principal direction vectors
- Feature loading table

Large absolute loading values indicate stronger alignment between a feature and a principal component.

### Projection & Reduction

Displays:

- Projected student scores
- Reduced representation
- Selected number of dimensions
- Percentage of retained variance

The projection formula is:

```text
Z = Xc Vk
```

### Visualizations

Displays:

1. Scree plot
2. Individual and cumulative explained variance plot
3. PC1 versus PC2 student scatter plot
4. Feature loading chart
5. Covariance heatmap in the covariance tab

### Verification

Compares the explicit manual PCA implementation with scikit-learn PCA.

The comparison includes:

- Eigenvalues versus scikit-learn explained variance
- Explained variance ratios
- Absolute numerical differences
- Projected score differences after sign alignment

Eigenvectors can have opposite signs while representing the same mathematical direction. Therefore, score signs are aligned before comparison.

### Mathematical Explanation

Provides expandable explanations of:

- Matrices
- Vector spaces
- Means and centering
- Covariance
- Eigenvalues and eigenvectors
- Orthogonality
- Projection
- Dimensionality reduction

It also provides factual numerical interpretation of the strongest feature loadings on PC1 and PC2.

## 9. CSV requirements

A custom CSV should contain:

- At least two numerical academic feature columns
- One row per student
- Optional identifier columns

Example:

```csv
Student Name,Roll Number,Mathematics,Physics,Programming,Attendance
Aarav Sharma,101,88,84,92,96
Diya Patel,102,76,79,85,91
Kabir Singh,103,92,90,87,94
```

Do not use student names or roll numbers as academic features.

## 10. Troubleshooting

### `python` command is not found

Install Python and ensure it is added to the system PATH. On some Linux/macOS systems, use `python3` and `pip3`.

### Packages are missing

Activate the virtual environment and reinstall:

```bash
python -m pip install -r requirements.txt
```

### Streamlit is not found

Run Streamlit through Python:

```bash
python -m streamlit run app.py
```

### Port 8501 is already in use

Use another port:

```bash
streamlit run app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

### The uploaded CSV is rejected

Check that:

- The file is a valid comma-separated CSV.
- It contains at least two numerical columns.
- Academic values are numeric rather than text.
- Column headers are present.

### The application shows missing-value warnings

This is expected when numerical cells are blank. The application fills missing numerical values using the relevant feature mean and reports the affected columns.

## 11. Recommended demonstration sequence

For a university presentation or viva:

1. Start the application with the bundled sample dataset.
2. Show the original student-feature matrix.
3. Explain why names and roll numbers are excluded.
4. Open the Centering tab and explain the feature means.
5. Show the covariance matrix and heatmap.
6. Explain eigenvalues and eigenvectors.
7. Demonstrate `VᵀV ≈ I`.
8. Select two or three principal components.
9. Show the loading table.
10. Explain the projection score matrix.
11. Show the retained variance after reduction.
12. Present the scree plot and PC1–PC2 scatter plot.
13. Open Verification and show agreement with scikit-learn.
14. Explain that PCA describes numerical variation and does not predict pass/fail outcomes.

## 12. Project maintenance

After changing the source code:

```bash
python -m pytest
python -m py_compile app.py src/*.py tests/test_pca.py
streamlit run app.py
```

Keep the following files together:

```text
app.py
requirements.txt
data/
src/
tests/
```

The application does not require a database, Firebase, login system, or external API.
