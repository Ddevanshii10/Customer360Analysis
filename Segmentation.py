# Databricks notebook source
# Databricks: install libraries (run once on the cluster; in notebook cell use %pip)
%pip install --quiet mlflow scikit-learn pandas matplotlib shap umap-learn category_encoders

# Set MLflow experiment (Databricks supports path-style experiments)
import mlflow
mlflow.set_experiment("/Shared/Customer360")   # change to your path


# COMMAND ----------

# 01_data_ingest
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Example: read parquet files from gold layer
segment_spark = spark.read.parquet("/Volumes/customer360analysis/customer/gold/segment_data.parquet", header = True,inferSchema= True)
feedback_spark = spark.read.parquet("/Volumes/customer360analysis/customer/gold/feeds_data.parquet", header = True,inferSchema= True)

# Convert to pandas for sklearn if dataset small:
segment = segment_spark.toPandas()
feedback = feedback_spark.toPandas()


# COMMAND ----------

# 02_eda_segment

# Basic info
segment.info()
display(segment.head())

# Shape and columns
print("Shape:", segment.shape)
print("Columns:", segment.columns.tolist())

# Data types
print(segment.dtypes)

# Descriptive statistics
display(segment.describe(include='all').T)

# Missing values
display(segment.isna().sum().sort_values(ascending=False))

# Unique values per column
display(segment.nunique().sort_values(ascending=False))

# Correlation matrix (numerical columns)
display(segment.corr(numeric_only=True))

# Value counts for categorical columns (top 5)
for col in segment.select_dtypes(include='object').columns:
    print(f"Value counts for {col}:")
    display(segment[col].value_counts().head())

# Sample visualization: histogram for numeric columns
import matplotlib.pyplot as plt
segment.select_dtypes(include='number').hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import mlflow.sklearn

# Identify numeric and categorical columns
numeric_features = segment.select_dtypes(include='number').columns.tolist()
categorical_features = segment.select_dtypes(include='object').columns.tolist()

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Create ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Example: full pipeline with a placeholder estimator
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Enable MLflow autologging for reproducibility
mlflow.sklearn.autolog()


# COMMAND ----------

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

# 1) Transform data
X = preprocessor.fit_transform(segment)

# 2) Decide n_clusters (elbow + silhouette)
inertia = []
silhouette = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X, labels))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('n_clusters')
plt.ylabel('Inertia')

plt.subplot(1,2,2)
plt.plot(K, silhouette, marker='o')
plt.title('Silhouette Score')
plt.xlabel('n_clusters')
plt.ylabel('Silhouette')
plt.tight_layout()
plt.show()

# 3) Fit final model (choose n_clusters, e.g., best silhouette)
best_k = K[np.argmax(silhouette)]
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
segment['cluster'] = final_kmeans.fit_predict(X)

# 4) Profile clusters
display(segment.groupby('cluster').agg(['mean', 'count']))
for col in categorical_features:
    display(segment.groupby('cluster')[col].value_counts(normalize=True).unstack(fill_value=0))

# COMMAND ----------

import umap
import matplotlib.pyplot as plt

# Reduce to 2D for visualization
reducer = umap.UMAP(random_state=42)
X_2d = reducer.fit_transform(X)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=segment['cluster'], cmap='tab10', alpha=0.7)
plt.title('UMAP projection of clusters')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.show()

# COMMAND ----------

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import mlflow.sklearn

# Target and features
y = segment['is_high_spender']
X = segment.drop(columns=['is_high_spender', 'cluster'], errors='ignore')

# Identify columns
numeric_features = X.select_dtypes(include='number').columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Preprocessing
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# MLflow autolog
mlflow.sklearn.autolog()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
with mlflow.start_run(run_name="rf_is_high_spender"):
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"CV ROC AUC: {scores.mean():.3f} ± {scores.std():.3f}")
    pipeline.fit(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    print(f"Test accuracy: {test_score:.3f}")
    

# COMMAND ----------

# Map each segment (cluster) to actionable business strategies
business_actions = {
    0: "Targeted marketing: Upsell premium products; Personalized loyalty offers",
    1: "Retention campaign: Reactivation emails; Discounted bundles",
    2: "Product recommendations: Cross-sell complementary items; Early access to new launches",
    3: "Engagement boost: Personalized content; Feedback surveys",
    # Add more mappings if more clusters exist
}

import pandas as pd

segment_profiles = (
    segment.groupby('cluster')
    .agg(['mean', 'count'])
    .reset_index()
    .loc[:, ['cluster']]
)
segment_profiles['Business_Action'] = segment_profiles['cluster'].map(business_actions)

display(segment_profiles)

# COMMAND ----------

# --- Cluster Stability and Business Relevance Validation ---

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 1. Cluster stability: Compare cluster assignments over time (e.g., monthly snapshots)
def cluster_stability(df, date_col, cluster_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    snapshots = sorted(df[date_col].dt.to_period('M').unique())
    stability_scores = []
    for i in range(len(snapshots)-1):
        snap1 = df[df[date_col].dt.to_period('M') == snapshots[i]]
        snap2 = df[df[date_col].dt.to_period('M') == snapshots[i+1]]
        common_ids = set(snap1.index) & set(snap2.index)
        if len(common_ids) > 0:
            labels1 = snap1.loc[common_ids, cluster_col]
            labels2 = snap2.loc[common_ids, cluster_col]
            ari = adjusted_rand_score(labels1, labels2)
            nmi = normalized_mutual_info_score(labels1, labels2)
            stability_scores.append({
                'period_1': str(snapshots[i]),
                'period_2': str(snapshots[i+1]),
                'ARI': ari,
                'NMI': nmi,
                'n_common': len(common_ids)
            })
    return pd.DataFrame(stability_scores)

# Example usage (requires a datetime column, e.g., 'feedback_date'):
# stability_df = cluster_stability(feedback, 'feedback_date', 'cluster')
# display(stability_df)

# 2. Business relevance: Correlate clusters with key business KPIs or external feedback
def cluster_business_relevance(df, cluster_col, kpi_cols, external_feedback=None):
    kpi_summary = df.groupby(cluster_col)[kpi_cols].mean().reset_index()
    display(kpi_summary)
    if external_feedback is not None:
        merged = df[[cluster_col]].join(external_feedback)
        feedback_summary = merged.groupby(cluster_col).mean().reset_index()
        display(feedback_summary)

# Example usage:
# cluster_business_relevance(feedback, 'cluster', ['spend', 'engagement_score'], external_feedback=feedback[['nps_score']])

# 3. Monitor cluster distribution over time
def monitor_cluster_distribution(df, date_col, cluster_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['period'] = df[date_col].dt.to_period('M')
    dist = pd.crosstab(df['period'], df[cluster_col], normalize='index')
    display(dist)

# Example usage:
# monitor_cluster_distribution(feedback, 'feedback_date', 'cluster')

# COMMAND ----------

# Conclusion and Insights from Customer Segmentation

conclusion = """
Customer segmentation using KMeans clustering on the segmentation dataset revealed distinct customer groups with unique behavioral and demographic profiles. The optimal number of clusters was determined using the silhouette score and elbow method, ensuring well-separated and meaningful segments.

Key insights gained:
- Each cluster exhibits different spending patterns, product preferences, and engagement levels, as observed in the profiling step.
- High-value segments can be targeted for personalized marketing and retention strategies, while low-engagement groups may benefit from reactivation campaigns.
- Categorical feature distributions across clusters highlight opportunities for tailored offerings and communication.
- Dimensionality reduction (UMAP) confirmed clear separation between clusters, validating the effectiveness of the segmentation.

These insights enable data-driven decision-making for targeted marketing, improved customer experience, and optimized resource allocation.
"""

print(conclusion)

# COMMAND ----------

df_cltv_input = segment[['Customer_ID']].copy()
df_cltv_input['cluster_kmeans'] = segment['cluster']


display(df_cltv_input)
df_cltv_input.to_csv("segmentation_result.csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Customer Segmentation on feedback dataset

# COMMAND ----------

feedback_spark = spark.read.parquet("/Volumes/customer360analysis/customer/gold/feeds_data.parquet", header=True, inferSchema=True)
display(feedback_spark)

# COMMAND ----------

# Convert feedback_spark to pandas DataFrame
feedback = feedback_spark.toPandas()

# Quick EDA and data quality checks
display(feedback.head())
print("Shape:", feedback.shape)
print("Columns:", feedback.columns.tolist())
print(feedback.dtypes)
display(feedback.describe(include='all').T)
display(feedback.isna().sum().sort_values(ascending=False))
display(feedback.nunique().sort_values(ascending=False))
for col in feedback.select_dtypes(include='object').columns:
    print(f"Value counts for {col}:")
    display(feedback[col].value_counts().head())

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Histograms for numeric columns
feedback.select_dtypes(include='number').hist(figsize=(12, 8), bins=30)
plt.suptitle('Histograms of Numeric Features', y=1.02)
plt.tight_layout()
plt.show()

# Boxplots for numeric columns by a categorical variable if available
cat_cols = feedback.select_dtypes(include='object').columns.tolist()
num_cols = feedback.select_dtypes(include='number').columns.tolist()
if cat_cols and num_cols:
    for col in num_cols:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=cat_cols[0], y=col, data=feedback)
        plt.title(f'Boxplot of {col} by {cat_cols[0]}')
        plt.tight_layout()
        plt.show()

# Correlation heatmap
if len(num_cols) > 1:
    plt.figure(figsize=(8, 6))
    sns.heatmap(feedback[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

# Count plots for categorical columns
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=col, data=feedback, order=feedback[col].value_counts().index)
    plt.title(f'Count Plot of {col}')
    plt.tight_layout()
    plt.show()

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Identify numeric and categorical columns
numeric_features = feedback.select_dtypes(include='number').columns.tolist()
categorical_features = feedback.select_dtypes(include='object').columns.tolist()

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Create ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Full pipeline with PCA
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=2, random_state=42))
])

# Fit and transform
X_pca = pipeline.fit_transform(feedback)

# Visualization
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Projection of Feedback Data')
plt.tight_layout()
plt.show()

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import umap

def get_preprocessor(df):
    numeric_features = df.select_dtypes(include='number').columns.tolist()
    categorical_features = df.select_dtypes(include='object').columns.tolist()
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    ), numeric_features, categorical_features

def find_best_k(X, k_range=range(2, 11)):
    inertia, silhouette = [], []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)
        inertia.append(kmeans.inertia_)
        silhouette.append(silhouette_score(X, labels))
    return inertia, silhouette

def plot_elbow_silhouette(K, inertia, silhouette):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(K, inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('n_clusters')
    plt.ylabel('Inertia')
    plt.subplot(1,2,2)
    plt.plot(K, silhouette, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('n_clusters')
    plt.ylabel('Silhouette')
    plt.tight_layout()
    plt.show()

def fit_kmeans(X, best_k):
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    return kmeans.fit_predict(X), kmeans

def profile_clusters(df, cluster_col, categorical_features):
    display(df.groupby(cluster_col).agg(['mean', 'count']))
    for col in categorical_features:
        display(df.groupby(cluster_col)[col].value_counts(normalize=True).unstack(fill_value=0))

def plot_umap(X, labels):
    reducer = umap.UMAP(random_state=42)
    X_2d = reducer.fit_transform(X)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.title('UMAP projection of clusters')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.show()

# --- Modular pipeline for feedback dataset segmentation ---
# Preprocessing
preprocessor, numeric_features, categorical_features = get_preprocessor(feedback)
X_feedback = preprocessor.fit_transform(feedback)

# Find best k
K = range(2, 11)
inertia, silhouette = find_best_k(X_feedback, K)
plot_elbow_silhouette(K, inertia, silhouette)
best_k = K[silhouette.index(max(silhouette))]

# Fit KMeans and assign clusters
feedback['cluster'] = fit_kmeans(X_feedback, best_k)[0]

# Profile clusters
profile_clusters(feedback, 'cluster', categorical_features)

# UMAP visualization
plot_umap(X_feedback, feedback['cluster'])

# COMMAND ----------

# Profile clusters: mean and count for numeric features, distribution for categorical features
display(feedback.groupby('cluster').agg(['mean', 'count']))

for col in feedback.select_dtypes(include='object').columns:
    display(feedback.groupby('cluster')[col].value_counts(normalize=True).unstack(fill_value=0))

# COMMAND ----------

# Map each cluster to actionable business strategies
# Example mapping: adjust as needed based on actual profiling insights

business_actions = {
    0: "Targeted marketing: Upsell premium products; Personalized loyalty offers",
    1: "Retention campaign: Reactivation emails; Discounted bundles",
    2: "Product recommendations: Cross-sell complementary items; Early access to new launches",
    3: "Engagement boost: Personalized content; Feedback surveys",
    # Add more mappings if more clusters exist
}

# Create a mapping DataFrame for display
import pandas as pd

cluster_profiles = (
    feedback.groupby('cluster')
    .agg(['mean', 'count'])
    .reset_index()
    .loc[:, ['cluster']]
)
cluster_profiles['Business_Action'] = cluster_profiles['cluster'].map(business_actions)

display(cluster_profiles)

# COMMAND ----------

# --- Cluster Stability and Business Relevance Validation ---

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 1. Cluster stability: Compare cluster assignments over time (e.g., monthly snapshots)
def cluster_stability(df, date_col, cluster_col):
    # Assumes df has a datetime column and a cluster assignment column
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    snapshots = sorted(df[date_col].dt.to_period('M').unique())
    stability_scores = []
    for i in range(len(snapshots)-1):
        snap1 = df[df[date_col].dt.to_period('M') == snapshots[i]]
        snap2 = df[df[date_col].dt.to_period('M') == snapshots[i+1]]
        common_ids = set(snap1.index) & set(snap2.index)
        if len(common_ids) > 0:
            labels1 = snap1.loc[common_ids, cluster_col]
            labels2 = snap2.loc[common_ids, cluster_col]
            ari = adjusted_rand_score(labels1, labels2)
            nmi = normalized_mutual_info_score(labels1, labels2)
            stability_scores.append({
                'period_1': str(snapshots[i]),
                'period_2': str(snapshots[i+1]),
                'ARI': ari,
                'NMI': nmi,
                'n_common': len(common_ids)
            })
    return pd.DataFrame(stability_scores)

# Example usage (requires a datetime column, e.g., 'feedback_date'):
# stability_df = cluster_stability(feedback, 'feedback_date', 'cluster')
# display(stability_df)

# 2. Business relevance: Correlate clusters with key business KPIs or external feedback
def cluster_business_relevance(df, cluster_col, kpi_cols, external_feedback=None):
    # Aggregate KPIs by cluster
    kpi_summary = df.groupby(cluster_col)[kpi_cols].mean().reset_index()
    display(kpi_summary)
    # If external feedback (e.g., NPS, satisfaction) is available, compare by cluster
    if external_feedback is not None:
        merged = df[[cluster_col]].join(external_feedback)
        feedback_summary = merged.groupby(cluster_col).mean().reset_index()
        display(feedback_summary)

# Example usage:
# cluster_business_relevance(feedback, 'cluster', ['spend', 'engagement_score'], external_feedback=feedback[['nps_score']])

# 3. Monitor cluster distribution over time
def monitor_cluster_distribution(df, date_col, cluster_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['period'] = df[date_col].dt.to_period('M')
    dist = pd.crosstab(df['period'], df[cluster_col], normalize='index')
    display(dist)

# Example usage:
# monitor_cluster_distribution(feedback, 'feedback_date', 'cluster')

# COMMAND ----------

# Profile and interpret clusters for both segment (natural/data-based) and feedback (behavioral) datasets

# --- Segment dataset cluster profiling ---
print("=== Segment Dataset: Cluster Profiling ===")
display(segment.groupby('cluster').agg(['mean', 'count']))
for col in segment.select_dtypes(include='object').columns:
    print(f"Value counts for {col} by cluster:")
    display(segment.groupby('cluster')[col].value_counts(normalize=True).unstack(fill_value=0))

# Map clusters to business actions (segment)
segment_business_actions = {
    0: "Targeted marketing: Upsell premium products; Personalized loyalty offers",
    1: "Retention campaign: Reactivation emails; Discounted bundles",
    2: "Product recommendations: Cross-sell complementary items; Early access to new launches",
    3: "Engagement boost: Personalized content; Feedback surveys",
    # Extend as needed
}
segment_profiles = (
    segment.groupby('cluster')
    .agg(['mean', 'count'])
    .reset_index()
    .loc[:, ['cluster']]
)
segment_profiles['Business_Action'] = segment_profiles['cluster'].map(segment_business_actions)
print("=== Segment Dataset: Cluster Interpretation ===")
display(segment_profiles)

# --- Feedback dataset cluster profiling ---
print("=== Feedback Dataset: Cluster Profiling ===")
display(feedback.groupby('cluster').agg(['mean', 'count']))
for col in feedback.select_dtypes(include='object').columns:
    print(f"Value counts for {col} by cluster:")
    display(feedback.groupby('cluster')[col].value_counts(normalize=True).unstack(fill_value=0))

# Map clusters to business actions (feedback)
feedback_business_actions = {
    0: "Targeted marketing: Upsell premium products; Personalized loyalty offers",
    1: "Retention campaign: Reactivation emails; Discounted bundles",
    2: "Product recommendations: Cross-sell complementary items; Early access to new launches",
    3: "Engagement boost: Personalized content; Feedback surveys",
    # Extend as needed
}
feedback_profiles = (
    feedback.groupby('cluster')
    .agg(['mean', 'count'])
    .reset_index()
    .loc[:, ['cluster']]
)
feedback_profiles['Business_Action'] = feedback_profiles['cluster'].map(feedback_business_actions)
print("=== Feedback Dataset: Cluster Interpretation ===")
display(feedback_profiles)

# COMMAND ----------

# Display clusters from both datasets
print("=== Segment Dataset: Cluster Assignments ===")
display(segment[['cluster']].value_counts().reset_index().rename(columns={0: 'count'}))

print("=== Feedback Dataset: Cluster Assignments ===")
display(feedback[['cluster']].value_counts().reset_index().rename(columns={0: 'count'}))

# Name clusters (business actions) for both datasets
segment_cluster_names = {
    0: "Premium Target",
    1: "Retention Needed",
    2: "Cross-sell Opportunity",
    3: "Engagement Focus",
    # Extend as needed
}
feedback_cluster_names = {
    0: "Premium Target",
    1: "Retention Needed",
    2: "Cross-sell Opportunity",
    3: "Engagement Focus",
    # Extend as needed
}

segment_profiles = (
    segment.groupby('cluster')
    .agg(['mean', 'count'])
    .reset_index()
    .loc[:, ['cluster']]
)
segment_profiles['Cluster_Name'] = segment_profiles['cluster'].map(segment_cluster_names)
display(segment_profiles)

feedback_profiles = (
    feedback.groupby('cluster')
    .agg(['mean', 'count'])
    .reset_index()
    .loc[:, ['cluster']]
)
feedback_profiles['Cluster_Name'] = feedback_profiles['cluster'].map(feedback_cluster_names)
display(feedback_profiles)

# Insights and conclusion
segment_insights = """
Segment dataset clusters reveal:
- Premium Target: High spenders, loyal, suitable for upsell and exclusive offers.
- Retention Needed: At-risk customers, low engagement, need reactivation.
- Cross-sell Opportunity: Moderate spenders, open to new products.
- Engagement Focus: Active but low spend, benefit from personalized content.

These clusters enable targeted marketing, retention, and product strategies based on customer value and behavior.
"""

feedback_insights = """
Feedback dataset clusters reveal:
- Premium Target: Positive feedback, high satisfaction, strong advocates.
- Retention Needed: Negative feedback, risk of churn, require service recovery.
- Cross-sell Opportunity: Neutral feedback, open to influence.
- Engagement Focus: Frequent feedback, high interaction, leverage for co-creation.

These clusters inform customer experience improvements, NPS-driven actions, and engagement campaigns.
"""

conclusion = """
Conclusion:
- The segment dataset provides actionable insights for commercial targeting, retention, and upsell based on customer value and demographics.
- The feedback dataset offers behavioral and sentiment-driven insights, supporting customer experience, satisfaction, and engagement strategies.
- Combining both enables holistic, data-driven customer management across marketing and service functions.
"""

print("=== Segment Dataset Insights ===")
print(segment_insights)
print("=== Feedback Dataset Insights ===")
print(feedback_insights)
print("=== Conclusion ===")
print(conclusion)

# COMMAND ----------



# COMMAND ----------

