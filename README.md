Customer Segmentation Web Application 📊

This is a user-friendly Flask-based web application designed to help businesses understand their customer base through automated segmentation. Users can upload their customer data, and the application will dynamically identify distinct customer groups, providing valuable insights for targeted marketing and strategic decision-making.

Key Features
Effortless CSV Upload: Simple interface for uploading customer datasets.

Dynamic Optimal K Determination: Automatically identifies the most appropriate number of customer segments (K) using the Silhouette Score, ensuring data-driven clustering.

Automated Data Processing: Handles necessary data scaling and encoding behind the scenes.

Insightful Visualizations: Generates a suite of plots including:

Elbow Method Plot

Silhouette Score Plot

Histograms for key customer attributes (Age, Annual Income, Spending Score)

Scatter plots illustrating customer segments based on spending, income, and age.

Actionable Segment Summaries: Provides tabular overviews of each cluster's average characteristics and gender distribution, facilitating easy interpretation.

Segmented Data Download: Allows users to download their original dataset augmented with the new cluster assignments.

How It Works
Upload Data: Users upload a CSV file via the web interface.

Define Columns: Users specify which columns in their CSV correspond to "Age," "Annual Income," "Spending Score," and "Gender."

Process & Cluster: The application preprocesses the data and applies the K-Means algorithm. It internally evaluates a range of K values to find the statistically optimal number of clusters for the uploaded data.

Visualize Results: Segmentation results, including plots and statistical summaries, are displayed on the results page.

Download: The processed dataset, now including cluster labels, can be downloaded.

Dataset Requirements
Your CSV file must include columns for Age, Annual Income, Spending Score, and Gender. The exact column names can vary, as you will specify them during the upload process.

Example Column Headers:
Age, Annual Income (k$), Spending Score (1-100), Genre

Getting Started (Local Setup)
To run this application on your local machine, follow these steps:

Prerequisites
Python 3.8+

pip (Python package installer)

Installation
Clone the repository:

Bash

git clone https://github.com/AmaarDevelops/Customer-Segmentation-Model-App.git
cd Customer-Segmentation-App
(Replace your-username with your actual GitHub username if you've forked or cloned your own repo.)

Create and activate a virtual environment (recommended):


python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install dependencies:

Ensure necessary directories exist:
Confirm templates/ and uploads/ directories are present in your project root. uploads will be created automatically by the app, but templates should contain index.html, results.html, and error.html.

Running the Application
Start the Flask development server:


python app.py
Access the application:
Open your web browser and navigate to http://127.0.0.1:5000/.

Project Structure
Customer-Segmentation-App/
├── app.py             # Main Flask application logic
├── model.py           # Contains core data preprocessing and clustering functions
├── requirements.txt   # Lists all Python package dependencies
├── templates/         # HTML template files for the web interface
│   ├── index.html     # Upload form and column specification
│   ├── results.html   # Displays analysis results and plots
│   └── error.html     # Generic error page
└── uploads/           # Directory for temporarily storing uploaded CSVs and generated image plots
Why This Application is Useful
This tool provides a powerful, yet accessible, solution for businesses seeking data-driven customer insights. It eliminates the need for specialized data science knowledge, allowing marketing teams and business analysts to quickly:

Tailor marketing campaigns to specific customer needs.

Personalize customer experiences based on segment characteristics.

Optimize resource allocation by focusing on the most relevant customer groups.

By automating a complex analytical process, the application makes advanced customer segmentation practical for a wider range of users.
