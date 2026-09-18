"""Matplotlib figures for PCA interpretation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scree_plot(eigenvalues: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    components = np.arange(1, len(eigenvalues) + 1)
    ax.plot(components, eigenvalues, marker="o", color="#2563eb", linewidth=2)
    ax.set(title="Scree Plot", xlabel="Principal Component", ylabel="Eigenvalue")
    ax.set_xticks(components)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def explained_variance_plot(ratios: np.ndarray, cumulative: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    components = np.arange(1, len(ratios) + 1)
    ax.bar(components, ratios * 100, color="#93c5fd", label="Individual")
    ax.plot(components, cumulative * 100, marker="o", color="#dc2626", label="Cumulative")
    ax.set(title="Explained Variance", xlabel="Principal Component", ylabel="Variance (%)")
    ax.set_xticks(components)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def pca_scatter(scores: np.ndarray, labels: pd.Series | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(scores[:, 0], scores[:, 1], s=70, color="#7c3aed", alpha=0.85)
    if labels is not None:
        for x, y, label in zip(scores[:, 0], scores[:, 1], labels.astype(str)):
            ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set(title="Students in the PC1-PC2 Plane", xlabel="PC1 score", ylabel="PC2 score")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def loading_plot(loadings: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    loadings.loc[:, loadings.columns[:2]].plot(kind="bar", ax=ax, color=["#2563eb", "#f97316"])
    ax.set(title="Feature Contributions to PC1 and PC2", xlabel="Academic Feature", ylabel="Loading")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Component")
    fig.tight_layout()
    return fig


def covariance_heatmap(covariance: np.ndarray, features: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(covariance, cmap="coolwarm")
    ax.set_xticks(range(len(features)), features, rotation=45, ha="right")
    ax.set_yticks(range(len(features)), features)
    ax.set_title("Covariance Matrix Heatmap")
    fig.colorbar(image, ax=ax, label="Covariance")
    fig.tight_layout()
    return fig
