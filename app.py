import os
import uuid
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template, send_from_directory


matplotlib.use('Agg')

from model import (
    load_and_inspect_data,
    preprocess_data,
    evaluate_kmeans_clusters,
    apply_final_clustering,
)

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Helper function to save all plots ---
# This function now takes NUMERICAL_FEATURES and CATEGORICAL_FEATURES
# to correctly reference column names for plotting.
def save_all_segmentation_plots(df: pd.DataFrame, upload_folder: str,
                                 k_range: range, inertia_scores: list,
                                 silhouette_values: list,
                                 numerical_features: list, categorical_features: list) -> list: # ADDED PARAMETERS
    """
    Generates and saves all required plots (Elbow, Silhouette, Histograms, Scatter plots)
    to the specified folder and returns their filenames.
    """
    plot_filenames = []

    # Unpack the specific feature names for clarity in plotting
    age_col = numerical_features[0] # 'Age'
    annual_income_col = numerical_features[1] 
    spending_score_col = numerical_features[2] 
    gender_col = categorical_features[0] 

    # 1. Elbow Method Plot (Using data returned from model.py)
    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), inertia_scores, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.grid(True)
    elbow_plot_name = f"elbow_method_{uuid.uuid4()}.png"
    plt.savefig(os.path.join(upload_folder, elbow_plot_name))
    plt.close()
    plot_filenames.append(elbow_plot_name)

    # 2. Silhouette Score Plot (Using data returned from model.py)
    plt.figure(figsize=(8, 6))
    # k_range[0:] because silhouette scores are not calculated for k=1 (or if k_range starts from 2)
    plt.plot(list(k_range)[0:], silhouette_values, marker='o', color='orange')
    plt.title('Silhouette Scores for Different K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    silhouette_plot_name = f"silhouette_score_{uuid.uuid4()}.png"
    plt.savefig(os.path.join(upload_folder, silhouette_plot_name))
    plt.close()
    plot_filenames.append(silhouette_plot_name)

    # 3. Histograms (Combined into one figure)
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    sns.histplot(data=df, x=annual_income_col, kde=True, bins=20) # CORRECTED
    plt.title('Distribution of Annual Income')
    plt.xlabel('Annual Income ($k)')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    sns.histplot(data=df, x=spending_score_col, kde=True, bins=20, color='orange') # CORRECTED
    plt.title('Distribution of Spending Score')
    plt.xlabel('Spending Score (1-100)')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    sns.histplot(data=df, x=age_col, kde=True, bins=20, color='purple') # CORRECTED
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    histograms_plot_name = f"histograms_{uuid.uuid4()}.png"
    plt.savefig(os.path.join(upload_folder, histograms_plot_name))
    plt.close()
    plot_filenames.append(histograms_plot_name)

    # 4. Scatter plot: Spending Score vs. Annual Income (raw cluster IDs)
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=df, x=spending_score_col, y=annual_income_col, hue='cluster', palette='viridis', s=80, alpha=0.8) # CORRECTED
    plt.title('Customer Segments: Spending Score vs. Annual Income (Cluster IDs)')
    plt.xlabel('Spending Score (1-100)')
    plt.ylabel('Annual Income ($k)')
    plt.legend(title='Cluster ID')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    scatter_raw_clusters_plot_name = f"spending_income_raw_clusters_{uuid.uuid4()}.png"
    plt.savefig(os.path.join(upload_folder, scatter_raw_clusters_plot_name))
    plt.close()
    plot_filenames.append(scatter_raw_clusters_plot_name)

    # 5. Scatter Plot with Labeled Clusters: Annual Income vs. Spending Score
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x=annual_income_col, y=spending_score_col, hue='Cluster_label', palette='tab10', s=80, alpha=0.8) # CORRECTED
    plt.title('Customer Segments by Annual Income vs. Spending Score (Labeled)')
    plt.xlabel('Annual Income ($k)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend(title='Customer Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    income_spending_labeled_plot_name = f"income_spending_labeled_segments_{uuid.uuid4()}.png"
    plt.savefig(os.path.join(upload_folder, income_spending_labeled_plot_name))
    plt.close()
    plot_filenames.append(income_spending_labeled_plot_name)

    # 6. Scatter Plot with Labeled Clusters: Age vs. Spending Score
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x=age_col, y=spending_score_col, hue='Cluster_label', palette='tab10', s=80, alpha=0.8) # CORRECTED
    plt.title('Customer Segments by Age vs. Spending Score (Labeled)')
    plt.xlabel('Age')
    plt.ylabel('Spending Score (1-100)')
    plt.legend(title='Customer Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    age_spending_labeled_plot_name = f"age_spending_labeled_segments_{uuid.uuid4()}.png"
    plt.savefig(os.path.join(upload_folder, age_spending_labeled_plot_name))
    plt.close()
    plot_filenames.append(age_spending_labeled_plot_name)

    return plot_filenames


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == "":
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())
    temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
    file.save(temp_filepath)

    try:
        # Get column names from user input
        age_col = request.form.get('age_col')
        income_col = request.form.get('income_col')
        spending_col = request.form.get('spending_col')
        gender_col = request.form.get('gender_col')

        
        NUMERICAL_FEATURES = [age_col, income_col, spending_col]
        CATEGORICAL_FEATURES = [gender_col]

        # 1. Load and Inspect Data
        df_original = load_and_inspect_data(temp_filepath)
        df_processing = df_original.copy()

        # 2. Preprocess Data
        x_preprocessed_data, preprocessor_obj = preprocess_data(
            df_processing, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
        )

        # Evaluate K-Means clusters for Elbow and Silhouette plots
        k_range = range(2, 11) # Start from 2 for Silhouette Score
        inertia_scores, silhouette_values = evaluate_kmeans_clusters(x_preprocessed_data, k_range)

        
        if not silhouette_values:
            optimal_k = k_range[0]
        else:
            best_silhouette_idx = silhouette_values.index(max(silhouette_values))
            optimal_k = list(k_range)[best_silhouette_idx]
        print(f"Dynamically determined OPTIMAL_K: {optimal_k}")
    

    
        df_clustered = apply_final_clustering(
            df_processing, x_preprocessed_data, optimal_k
        )

    
        df_clustered['Cluster_label'] = df_clustered['cluster'].apply(lambda x: f"Cluster {x}")

        # Generate and save all plots using the helper function
        # NOW PASSING THE FEATURE LISTS TO THE PLOTTING FUNCTION
        plot_filenames = save_all_segmentation_plots(
            df_clustered, app.config['UPLOAD_FOLDER'],
            k_range, inertia_scores, silhouette_values,
            NUMERICAL_FEATURES, CATEGORICAL_FEATURES # ADDED ARGUMENTS
        )

        # 4. Save segmented CSV
        segmented_output_filename = f"segmented_customers_{unique_id}.csv"
        segmented_output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], segmented_output_filename)
        df_clustered.to_csv(segmented_output_filepath, index=False)

        
        cluster_summary_html = df_clustered.groupby('Cluster_label')[NUMERICAL_FEATURES].mean().to_html()

    
        gender_distribution = df_clustered.groupby('Cluster_label')[CATEGORICAL_FEATURES[0]].value_counts(normalize=True).unstack(fill_value=0)
        gender_distribution_html = gender_distribution.to_html()

        
        return render_template(
            'results.html',
            cluster_count=optimal_k,
            download_link=segmented_output_filename,
            plot_urls=[url_for('download', filename=f) for f in plot_filenames],
            cluster_summary=cluster_summary_html,
            gender_distribution=gender_distribution_html,
        )

    except Exception as e:
        print(f"Error processing file: {e}")
        return render_template('error.html', message=f"Error processing your file: {e}")

    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)