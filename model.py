import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings from scikit-learn for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) # For potential matplotlib/tkinter issues

# --- Data Loading and Inspection ---
def load_and_inspect_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a CSV file and performs basic inspection.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {filepath}")
    except Exception as e:
        raise Exception(f"Error loading or inspecting data: {e}")

# --- Data Preprocessing ---
def preprocess_data(df: pd.DataFrame, numerical_features: list, categorical_features: list):
    """
    Preprocesses the data using StandardScaler for numerical features
    and OneHotEncoder for categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    X_preprocessed = preprocessor.fit_transform(df)

    return X_preprocessed, preprocessor

# --- K-Means Evaluation (Elbow and Silhouette) ---
def evaluate_kmeans_clusters(preprocessed_data: np.ndarray, k_range: range) -> tuple[list, list]:
    """
    Evaluates K-Means clustering for a range of K values using
    the Elbow Method (inertia) and Silhouette Score.
    """
    inertia_scores = []
    silhouette_values = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(preprocessed_data)
        inertia_scores.append(kmeans.inertia_)

        if k > 1: 
            score = silhouette_score(preprocessed_data, kmeans.labels_)
            silhouette_values.append(score)
        else: 
            silhouette_values.append(0) 

  
    return inertia_scores, silhouette_values


def apply_final_clustering(df_original: pd.DataFrame, preprocessed_data: np.ndarray, optimal_k: int) -> pd.DataFrame:
    """
    Applies K-Means clustering with the optimal K and returns the DataFrame
    with raw cluster assignments.
    """
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(preprocessed_data)
    df_original['cluster'] = clusters # Assign raw cluster ID (e.g., 0, 1, 2...)

    # 'Cluster_label' will now be added in app.py (e.g., 'Cluster 0', 'Cluster 1')
    return df_original

