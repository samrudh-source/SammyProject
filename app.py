"""Streamlit dashboard for the Academic Performance Principal Pattern Analyzer."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import (  # noqa: E402
    identify_pca_columns,
    load_sample_data,
    prepare_numeric_features,
    validate_dataset,
)
from src.pca_calculator import calculate_pca, component_loadings, orthogonality_matrix  # noqa: E402
from src.verification import verify_against_sklearn  # noqa: E402
from src.visualization import (  # noqa: E402
    covariance_heatmap,
    explained_variance_plot,
    loading_plot,
    pca_scatter,
    scree_plot,
)

st.set_page_config(page_title="Academic Performance PCA Analyzer", page_icon="📐", layout="wide")

FORMULAS = {
    "Mean": r"\mu_j = \frac{1}{n}\sum_{i=1}^{n} X_{ij}",
    "Centered matrix": r"X_c = X - \mu",
    "Covariance matrix": r"C = \frac{1}{n-1}X_c^T X_c",
    "Eigenvalue equation": r"Cv = \lambda v",
    "Orthogonality": r"v_i^T v_j = 0,\quad i \ne j",
    "Projection": r"Z = X_c V_k",
    "Explained variance ratio": r"EVR_i = \frac{\lambda_i}{\sum_j \lambda_j}",
    "Cumulative explained variance": r"CEV_k = \sum_{i=1}^{k} EVR_i",
}


def frame(values: np.ndarray, index=None, columns=None, precision: int = 4) -> pd.DataFrame:
    return pd.DataFrame(values, index=index, columns=columns).round(precision)


def get_dataset() -> pd.DataFrame:
    if "dataset" not in st.session_state:
        st.session_state.dataset = load_sample_data()
    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader("Upload student CSV", type=["csv"])
        if uploaded is not None:
            try:
                st.session_state.dataset = pd.read_csv(uploaded)
                st.success("Uploaded dataset loaded.")
            except Exception as error:
                st.error(f"Could not read CSV: {error}")
        col1, col2 = st.columns(2)
        if col1.button("Load Sample Dataset", use_container_width=True):
            st.session_state.dataset = load_sample_data()
            st.rerun()
        if col2.button("Reset", use_container_width=True):
            st.session_state.dataset = load_sample_data()
            st.session_state.pop("result", None)
            st.rerun()
    return st.session_state.dataset


def render_formula(name: str) -> None:
    st.latex(FORMULAS[name])


def describe_concept(title: str, text: str) -> None:
    with st.expander(title):
        st.write(text)


def main() -> None:
    st.title("Academic Performance Principal Pattern Analyzer")
    st.subheader("Linear Algebra Based Student Performance Analysis using PCA")
    st.caption("A transparent sequence: Data → Matrix → Centering → Covariance → Eigen analysis → Projection → Reduction → Interpretation")

    data = get_dataset()
    columns, warnings = validate_dataset(data)
    for warning in warnings:
        st.warning(warning)
    if len(columns) < 2:
        st.error("Please provide at least two numerical academic columns.")
        return
    features = prepare_numeric_features(data, columns)
    labels = data.get("Student Name", data.get("Student_Name", pd.Series(range(1, len(data) + 1))))

    with st.sidebar:
        st.metric("Students", len(features))
        st.metric("PCA features", len(columns))
        if st.button("Run PCA", type="primary", use_container_width=True):
            st.session_state.result = calculate_pca(features.to_numpy(dtype=float))
            st.success("PCA pipeline completed.")
    result = st.session_state.get("result")

    tabs = st.tabs(["Home", "Dataset", "Centering", "Covariance Matrix", "Eigen Analysis",
                    "Principal Components", "Projection & Reduction", "Visualizations",
                    "Verification", "Mathematical Explanation"])
    with tabs[0]:
        st.markdown("### Welcome")
        st.write("This dashboard computes PCA step by step. Every displayed stage is produced from the preceding stage, rather than from a single black-box PCA call.")
        st.info("Use the sample data immediately, or upload a CSV with numerical academic feature columns. Student names and roll numbers are retained only as labels.")
        st.markdown("#### Sequential mathematical chain")
        st.code("DATA\n  ↓\nSTUDENT-FEATURE MATRIX X\n  ↓\nCENTERED MATRIX X_c\n  ↓\nCOVARIANCE MATRIX C\n  ↓\nEIGENVALUES + EIGENVECTORS\n  ↓\nORTHOGONAL PRINCIPAL DIRECTIONS\n  ↓\nPROJECTION Z = X_c V_k\n  ↓\nREDUCED REPRESENTATION\n  ↓\nINTERPRETATION")
        describe_concept("What is PCA?", "Principal Component Analysis rotates a centered feature space into orthogonal directions ranked by variance. It helps summarize correlated academic measurements while retaining the most variation.")
    with tabs[1]:
        st.header("Stage 1 — Matrix Representation")
        st.write(f"X ∈ R^({len(features)}×{len(columns)}), where n is the number of students and m is the number of academic features.")
        st.dataframe(features, use_container_width=True)
        st.caption("X = Student-Feature Matrix (identifiers are excluded from PCA).")
    with tabs[2]:
        st.header("Stage 2 — Data Centering")
        if result is None:
            st.info("Click Run PCA in the sidebar to calculate the intermediate matrices.")
        else:
            render_formula("Mean")
            st.write("The mean vector gives the average of each academic feature. Centering moves the feature-space origin to this mean.")
            st.dataframe(frame(result.means.reshape(1, -1), columns=columns), use_container_width=True)
            st.write("Original matrix X")
            st.dataframe(frame(result.matrix, columns=columns), use_container_width=True)
            st.write("Centered matrix X_c = X − μ")
            st.dataframe(frame(result.centered_matrix, columns=columns), use_container_width=True)
            render_formula("Centered matrix")
    with tabs[3]:
        st.header("Stage 3 — Covariance Matrix")
        if result is not None:
            render_formula("Covariance matrix")
            st.dataframe(frame(result.covariance_matrix, index=columns, columns=columns), use_container_width=True)
            st.write("Each entry measures how two academic features vary together. The diagonal contains each feature's variance.")
            st.pyplot(covariance_heatmap(result.covariance_matrix, columns), clear_figure=True)
        else:
            st.info("Run PCA to calculate covariance from the centered matrix.")
    with tabs[4]:
        st.header("Stage 4 — Eigenvalue and Eigenvector Analysis")
        if result is not None:
            render_formula("Eigenvalue equation")
            eigen_table = pd.DataFrame({
                "Component": [f"PC{i}" for i in range(1, len(columns) + 1)],
                "Eigenvalue": result.eigenvalues,
                "Explained variance (%)": result.explained_variance_ratio * 100,
                "Cumulative (%)": result.cumulative_explained_variance * 100,
            }).round(4)
            st.dataframe(eigen_table, hide_index=True, use_container_width=True)
            st.write("Eigenvectors (columns are principal directions)")
            st.dataframe(frame(result.eigenvectors, index=columns,
                               columns=[f"PC{i}" for i in range(1, len(columns) + 1)]), use_container_width=True)
            st.write("Eigenvectors are directions; eigenvalues quantify the variance along those directions.")
        else:
            st.info("Run PCA to calculate eigenvalues and eigenvectors.")
    with tabs[5]:
        st.header("Stage 5–6 — Orthogonality and Principal Components")
        if result is not None:
            render_formula("Orthogonality")
            st.write("For normalized eigenvectors, VᵀV should be approximately the identity matrix.")
            st.dataframe(frame(orthogonality_matrix(result.eigenvectors),
                               columns=[f"PC{i}" for i in range(1, len(columns) + 1)]), use_container_width=True)
            components = st.slider("Number of principal components", 2, len(columns), min(3, len(columns)))
            loadings = component_loadings(result.eigenvectors, columns, components)
            st.write(f"Selected principal direction vectors: PC1 through PC{components}")
            st.dataframe(loadings.round(4), use_container_width=True)
            st.caption("Large absolute loadings indicate features aligned strongly with that principal direction. Sign indicates direction, while the overall sign of an eigenvector is arbitrary.")
        else:
            st.info("Run PCA to inspect orthogonality and choose principal components.")
    with tabs[6]:
        st.header("Stage 7–8 — Projection and Dimensionality Reduction")
        if result is not None:
            components = st.slider("Components retained for projection", 2, len(columns), min(2, len(columns)), key="projection_components")
            scores = result.project(components)
            render_formula("Projection")
            st.write(f"Z has shape {scores.shape}: each student is represented by {components} principal coordinates instead of {len(columns)} original features.")
            score_table = pd.DataFrame(scores, columns=[f"PC{i}" for i in range(1, components + 1)])
            score_table.insert(0, "Student", labels.astype(str).to_numpy())
            st.dataframe(score_table.round(4), use_container_width=True)
            render_formula("Explained variance ratio")
            retained = result.cumulative_explained_variance[components - 1]
            st.metric("Variance retained", f"{retained:.2%}", f"Reduced from {len(columns)}D to {components}D")
            st.write("The projected scores are the reduced representation. They preserve the selected directions' share of total variance.")
        else:
            st.info("Run PCA to project students into a reduced coordinate system.")
    with tabs[7]:
        st.header("Visualizations")
        if result is not None:
            left, right = st.columns(2)
            with left:
                st.pyplot(scree_plot(result.eigenvalues), clear_figure=True)
            with right:
                st.pyplot(explained_variance_plot(result.explained_variance_ratio, result.cumulative_explained_variance), clear_figure=True)
            scores = result.project(2)
            st.pyplot(pca_scatter(scores, labels), clear_figure=True)
            st.pyplot(loading_plot(component_loadings(result.eigenvectors, columns, min(2, len(columns)))), clear_figure=True)
        else:
            st.info("Run PCA to generate plots.")
    with tabs[8]:
        st.header("Verification with scikit-learn")
        if result is not None:
            comparison, _, aligned_scores = verify_against_sklearn(result)
            st.write("scikit-learn is used only as an independent reference. The application PCA above uses the explicit covariance/eigen-decomposition workflow.")
            st.dataframe(comparison.round(8), hide_index=True, use_container_width=True)
            st.write(f"Maximum absolute score difference after sign alignment: {np.max(np.abs(result.project(len(columns)) - aligned_scores)):.8f}")
            st.caption("Eigenvectors and scores may differ by sign because v and −v describe the same principal direction.")
        else:
            st.info("Run PCA to compare manual calculations with scikit-learn.")
    with tabs[9]:
        st.header("Mathematical Explanation")
        explanations = {
            "Matrix": "A matrix is a rectangular arrangement of values. X stores one student's feature vector per row.",
            "Vector space": "Each row is a point in an m-dimensional academic feature space; each column is a coordinate axis.",
            "Mean and centering": "Subtracting the feature mean translates the cloud of points so its center is the origin.",
            "Covariance": "The covariance matrix captures pairwise co-variation and variances of centered features.",
            "Eigenvalues/eigenvectors": "An eigenvector is a direction that covariance does not rotate; its eigenvalue is the variance on that direction.",
            "Orthogonality": "Orthogonal principal directions contain non-overlapping directional information and form a rotated coordinate system.",
            "Projection": "A dot product with a principal direction gives a student's coordinate (score) on that direction.",
            "Dimensionality reduction": "Keeping only the first k directions reduces coordinates while retaining the cumulative explained variance.",
        }
        for name, explanation in explanations.items():
            describe_concept(name, explanation)
        st.markdown("#### Formula reference")
        for name, formula in FORMULAS.items():
            st.write(name)
            st.latex(formula)
        st.markdown("#### Factual pattern interpretation")
        if result is not None:
            top_pc1 = sorted(zip(columns, np.abs(result.eigenvectors[:, 0])), key=lambda item: item[1], reverse=True)[:3]
            top_pc2 = sorted(zip(columns, np.abs(result.eigenvectors[:, 1])), key=lambda item: item[1], reverse=True)[:3]
            st.write(f"PC1 captures {result.explained_variance_ratio[0]:.2%} of variance; strongest absolute loadings: " + ", ".join(name for name, _ in top_pc1) + ".")
            st.write(f"PC2 captures {result.explained_variance_ratio[1]:.2%} of variance; strongest absolute loadings: " + ", ".join(name for name, _ in top_pc2) + ".")
            st.write("Students close together in the PC1–PC2 scatter have similar coordinates in the retained principal directions. This is a numerical similarity statement, not a prediction of success or failure.")


if __name__ == "__main__":
    main()
