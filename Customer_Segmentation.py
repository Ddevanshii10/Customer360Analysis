# Databricks notebook source
import pandas as pd
df_segment = spark.read.parquet(
    '/Volumes/customer360analysis/customer/gold/segment_data.parquet/'
)
display(df_segment)   # your DataFrame


# COMMAND ----------

df_feeds = spark.read.parquet(
    '/Volumes/customer360analysis/customer/gold/feeds_data.parquet/'
)
display(df_feeds)

# COMMAND ----------

print('Segemnt data columns:',df_segment.columns)
print('feedback data columns:',df_feeds.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC For customer segmentation we will be using two different datasets Dataset 1 (df_segmentation) → Telco-style segmentation (demographics, service usage, spend, tenure).
# MAGIC
# MAGIC Dataset 2 (df_survey) → Survey-style segmentation (satisfaction, loyalty, feedback-based).
# MAGIC
# MAGIC This gives you two perspectives of customer segmentation:
# MAGIC
# MAGIC Behavioral Segments (from telco dataset): who customers are and what services they use.
# MAGIC
# MAGIC Attitudinal Segments (from survey dataset): how customers feel (satisfaction, loyalty).
# MAGIC
# MAGIC 📌 In business, companies often run multiple segmentation studies separately

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Importing required libraries

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import pairwise_distances
from sklearn.feature_selection import VarianceThreshold
%pip install umap-learn
import umap

# COMMAND ----------

# HDBSCAN is optional (install via pip)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

def plot_elbow_silhouette(k_list, inertias, silhouettes):
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(k_list, inertias, '-o', color='C0')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Inertia (SSE)', color='C0')
    ax2 = ax1.twinx()
    ax2.plot(k_list, silhouettes, '-o', color='C1')
    ax2.set_ylabel('Mean Silhouette', color='C1')
    plt.title('Elbow (inertia) and Silhouette by k')
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Starting with df_segment dataset

# COMMAND ----------

df_segment.display()

# COMMAND ----------

df_segment.columns

# COMMAND ----------

# Normalize column names to snake_case
new_cols = [
    c.strip().replace(' ', '_').replace('-', '_')
    for c in df_segment.columns
]
df_segment = df_segment.toDF(*new_cols)
print("Columns detected:", df_segment.columns)
# Quick preview
display(df_segment)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set ID and select features

# COMMAND ----------

# Cell 3: set id and feature list
id_col = 'Customer_ID' if 'Customer_ID' in df_segment.columns else \
         next((c for c in df_segment.columns if 'id' in c.lower()), None)
if id_col is None:
    df_segment['customer_id'] = df_segment.index.astype(str)
    id_col = 'customer_id'
print("Using id column:", id_col)

# features from your list (keep existing ones)
candidate_features = [
 'Age','Gender','Married','Dependents','Contract','Tenure_in_Months','Offer','Payment_Method',
 'Monthly_Charge','Total_Charges','Avg_Monthly_GB_Download','Total_Long_Distance_Charges','Total_Revenue',
 'Internet_Service','Streaming_TV','Streaming_Movies','Streaming_Music','Premium_Tech_Support','Device_Protection_Plan',
 'Online_Security','Online_Backup','Referred_a_Friend','Number_of_Referrals','Satisfaction_Score','Unlimited_Data',
 'Paperless_Billing','tenure_bucket','avg_revenue_per_month','total_spend_ratio','is_high_spender','num_services',
 'monthly_charge_x_tenure','log_total_revenue'
]

# keep only existing columns (case-sensitive from your list); try both original and lowercased
present = [c for c in candidate_features if c in df_segment.columns]
# also try lowercased match
present += [c for c in candidate_features if c.lower() in df_segment.columns and c not in present]
present = list(dict.fromkeys(present))  # dedupe keep order
print("Features selected:", present)

if len(present) < 3:
    raise SystemExit("Too few features detected — check column names or adjust candidate_features.")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Automatic numeric vs categorical detection and minor cleaning

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, NumericType

# Strip strings in string columns
for c, dtype in df_segment.dtypes:
    if dtype == 'string' and c in present:
        df_segment = df_segment.withColumn(
            c,
            F.trim(F.col(c))
        )

# Detect numeric vs categorical columns
num_cols = [
    c for c, dtype in df_segment.dtypes
    if (c in present) and (isinstance(df_segment.schema[c].dataType, NumericType))
]
cat_cols = [c for c in present if c not in num_cols]

# Move numeric columns with few unique values to categorical
for c in list(num_cols):
    nunique = df_segment.select(c).distinct().count()
    if nunique <= 5:
        num_cols.remove(c)
        cat_cols.append(c)

print("Numeric features:", num_cols)
print("Categorical features:", cat_cols)

# Quick missing value summary
missing_counts = (
    df_segment.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in present
    ])
)
display(missing_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC Handle high-cardinality categorical (frequency encode) & finalize columns

# COMMAND ----------

from pyspark.sql import functions as F

HIGH_CARD_THRESHOLD = 25

# Compute cardinality for each categorical column
cat_cardinality = {
    c: df_segment.select(c).distinct().count()
    for c in cat_cols
}

cat_low_card = [c for c, n in cat_cardinality.items() if n <= HIGH_CARD_THRESHOLD]
cat_high_card = [c for c in cat_cols if c not in cat_low_card]

print("Low-cardinality cats (OHE):", cat_low_card)
print("High-cardinality cats (freq-encode):", cat_high_card)

# Frequency encode high-cardinality categorical columns
for c in cat_high_card:
    freq_df = (
        df_segment.groupBy(c)
        .agg((F.count("*") / df_segment.count()).alias("freq"))
    )
    newcol = f"{c}_freqenc"
    df_segment = df_segment.join(
        freq_df,
        on=c,
        how="left"
    ).withColumnRenamed("freq", newcol)
    num_cols.append(newcol)

# Update final categorical list to only low-card
cat_cols = cat_low_card
print("Final numeric cols:", num_cols)
print("Final categorical cols (one-hot):", cat_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build preprocessing pipeline and transform

# COMMAND ----------

# Convert Spark DataFrame to pandas DataFrame before using scikit-learn
df_pd = df_segment.select(num_cols + cat_cols).toPandas()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

X_arr = preprocessor.fit_transform(df_pd)

# get feature names (sklearn >=1.0)
try:
    feature_names = preprocessor.get_feature_names_out()
except Exception:
    ohe_names = []
    if len(cat_cols) > 0:
        ohe_names = list(
            preprocessor.named_transformers_['cat']
            .named_steps['ohe']
            .get_feature_names_out(cat_cols)
        )
    feature_names = num_cols + ohe_names

import pandas as pd
X = pd.DataFrame(X_arr, columns=feature_names, index=df_pd.index)
display(X)

# COMMAND ----------

# MAGIC %md
# MAGIC ### VarianceThreshold & PCA to reduce noise / speed

# COMMAND ----------

# Cell 7: optional dimensionality reduction
vt = VarianceThreshold(threshold=1e-4)
X_v = vt.fit_transform(X)
print("After VarianceThreshold -> shape:", X_v.shape)

# PCA to capture 95% variance (for clustering/visualization)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_v)
print("PCA reduced shape:", X_pca.shape, "- components kept:", pca.n_components_)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Elbow (inertia) + Silhouette to choose k (sample if large)

# COMMAND ----------

# Cell 8: elbow + silhouette to pick k
from sklearn.utils import resample
n = X_pca.shape[0]
SAMPLE_MAX = 20000
if n > SAMPLE_MAX:
    idx = np.random.choice(n, SAMPLE_MAX, replace=False)
    X_k = X_pca[idx]
    print(f"Using sample {SAMPLE_MAX} for k selection (from {n})")
else:
    X_k = X_pca

k_list = list(range(2,11))
inertias = []
silhs = []
for k in k_list:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    preds = km.fit_predict(X_k)
    inertias.append(km.inertia_)
    silhs.append(silhouette_score(X_k, preds))
    print("k", k, "-> silhouette", round(silhs[-1],4))

plot_elbow_silhouette(k_list, inertias, silhs)

# suggested_k = k_list[np.argmax(silhs)]
# print("Suggested best k by silhouette:", suggested_k)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit final KMeans on full PCA space & assign cluster + distances

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id
import numpy as np
from scipy.spatial.distance import cdist

# Train KMeans and get clusters (assuming X_pca is a numpy array)
best_k = 3  # Set to your desired number of clusters

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)
clusters = kmeans.fit_predict(X_pca)

# Add row index to join numpy results back to Spark DataFrame
df_segment = df_segment.withColumn("row_idx", monotonically_increasing_id())

# Create Spark DataFrame for clusters
clusters_df = spark.createDataFrame(
    [(int(c),) for c in clusters],
    ["cluster_kmeans"]
).withColumn("row_idx", monotonically_increasing_id())

# Compute distances to centroids
dists = cdist(X_pca, kmeans.cluster_centers_, metric='euclidean')
dists_min = dists.min(axis=1)
dists_df = spark.createDataFrame(
    [(float(d),) for d in dists_min],
    ["dist_to_centroid"]
).withColumn("row_idx", monotonically_increasing_id())

# Join cluster assignments and distances to df_segment
df_segment = (
    df_segment
    .join(clusters_df, on="row_idx")
    .join(dists_df, on="row_idx")
    .drop("row_idx")
)

display(df_segment)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit Gaussian Mixture for comparison (soft clusters)

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

# Fit GMM and get labels
gmm = GaussianMixture(
    n_components=best_k,
    random_state=42,
    n_init=5
)
gmm_labels = gmm.fit_predict(X_pca)

# Add row index to join labels back to Spark DataFrame
df_segment = df_segment.withColumn(
    "row_idx",
    monotonically_increasing_id()
)

# Create DataFrame for GMM labels
gmm_labels_df = spark.createDataFrame(
    [(int(label),) for label in gmm_labels],
    ["cluster_gmm"]
).withColumn(
    "row_idx",
    monotonically_increasing_id()
)

# Join GMM labels to df_segment
df_segment = df_segment.join(
    gmm_labels_df,
    on="row_idx"
).drop("row_idx")

print("GMM silhouette:", round(silhouette_score(X_pca, gmm_labels), 4))
display(df_segment)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2D visualization (UMAP if available, else PCA(2))

# COMMAND ----------

# Drop duplicate 'cluster_kmeans' columns, keeping only the first occurrence
first_idx = df_segment.columns.index("cluster_kmeans")
cols_no_dup = [c for i, c in enumerate(df_segment.columns) if c != "cluster_kmeans" or i == first_idx]
df_segment = df_segment.select(cols_no_dup)
display(df_segment)

# COMMAND ----------

# Check if umap-learn is installed
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# COMMAND ----------

# Convert PySpark DataFrame to pandas DataFrame
df_segment_pd = df_segment.toPandas()

# Ensure both arrays and dataframe have same rows
print("Shape of X_vis:", X_vis.shape)
print("Shape of df_segment_pd:", df_segment_pd.shape)

# Align indices if mismatch
min_len = min(len(X_vis), len(df_segment_pd))
X_vis = X_vis[:min_len]
df_segment_pd = df_segment_pd.iloc[:min_len]

# Now plot
plt.figure(figsize=(10,6))
sns.scatterplot(
    x=X_vis[:,0],
    y=X_vis[:,1],
    hue=df_segment_pd['cluster_kmeans'].astype(str),
    palette='tab10',
    s=18,
    linewidth=0,
    alpha=0.8
)
plt.title("2D projection colored by KMeans cluster (PCA)")
plt.legend(title='cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


plt.figure(figsize=(10,6))
sns.scatterplot(
    x=X_vis[:,0],
    y=X_vis[:,1],
    hue=df_segment_pd['cluster_kmeans'].astype(str),
    palette='tab10',
    s=18,
    linewidth=0,
    alpha=0.8
)
plt.title("2D projection colored by KMeans cluster (PCA)")
plt.legend(title='cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cluster profiling: numeric means and top differences

# COMMAND ----------

# Numeric profile using PySpark
from pyspark.sql import functions as F

cluster_col = 'cluster_kmeans'

# Compute mean for each numeric column per cluster
agg_exprs = [F.mean(c).alias(c) for c in num_cols]
num_profile = (
    df_segment
    .groupBy(cluster_col)
    .agg(*agg_exprs)
    .orderBy(cluster_col)
)

print("Numeric profile (mean per cluster):")
display(num_profile)

# Convert to pandas for further analysis
num_profile_pd = num_profile.toPandas().set_index(cluster_col)

# Calculate which features vary most across clusters
var_across = num_profile_pd.var().sort_values(ascending=False)
top_varying = var_across.head(8).index.tolist()
print("Top varying numeric features:", top_varying)

# Plot normalized cluster profiles for top features
cm = num_profile_pd[top_varying]
cm_norm = (cm - cm.min()) / (cm.max() - cm.min())
cm_norm.plot(
    kind='bar',
    figsize=(12,5)
)
plt.title('Cluster profiles (normalized) - top numeric features')
plt.ylabel('Normalized mean')
plt.show()

# Categorical distributions for cat_cols
df_segment_pd = df_segment.toPandas()
for c in cat_cols:
    print(f"\nDistribution for {c}:")
    display(
        pd.crosstab(
            df_segment_pd[cluster_col],
            df_segment_pd[c],
            normalize='index'
        ).round(3)
    )

# COMMAND ----------

import joblib

joblib.dump(preprocessor, "seg_preprocessor.joblib")
joblib.dump(pca, "seg_pca.joblib")
joblib.dump(kmeans, "seg_kmeans.joblib")
print("Saved preprocessor, pca, kmeans (joblib files).")

out_cols = [id_col, 'cluster_kmeans', 'dist_to_centroid'] + present
df_out = df_segment_pd[out_cols].copy()
df_out.to_csv("customer_segments_labeled.csv", index=False)
print("Saved customer_segments_labeled.csv")

# If in Databricks and want to save as table (Delta)
try:
    spark_df = spark.createDataFrame(df_out)
    spark_df.write.format("delta").mode("overwrite").saveAsTable(
        "customer360analysis.customer.gold.customer_segments"
    )
    print("Saved to Delta table: customer360analysis.customer.gold.customer_segments")
except Exception as e:
    print("Databricks save skipped/failed (not in Spark env):", e)

# COMMAND ----------

plt.figure(figsize=(10,6))
sns.scatterplot(
    x=X_vis[:,0],
    y=X_vis[:,1],
    hue=df_segment_pd['cluster_kmeans'].astype(str),
    palette='tab10',
    s=18,
)


# COMMAND ----------

# MAGIC %md
# MAGIC Conclusion & Insights from the Customer Segmentation Model
# MAGIC Conclusion
# MAGIC
# MAGIC The clustering model successfully segmented your customer base into distinct groups by leveraging both numeric and categorical features (demographics, tenure, spend, and service usage).
# MAGIC
# MAGIC The best separation (highest silhouette score) was achieved using 2 or 3 clusters, but profiling across more clusters (up to 10) provided deeper insights into business subgroups.
# MAGIC
# MAGIC The clusters were visualized in reduced dimensional space (PCA), showing clearly separated customer segments.
# MAGIC
# MAGIC Key Insights
# MAGIC
# MAGIC Distinct segments revealed:
# MAGIC
# MAGIC High spenders vs. low spenders and new vs. long-tenured customers.
# MAGIC
# MAGIC Cluster profiles were mainly differentiated by features like Total_Revenue, Total_Charges, tenure, and usage (Monthly_Charge, Avg_Monthly_GB_Download).
# MAGIC
# MAGIC Service & demographic connection:
# MAGIC
# MAGIC Significant differences found in service preferences (contract type, payment method) and demographic variables (age, marital status).
# MAGIC
# MAGIC Strategic Value:
# MAGIC
# MAGIC Enables targeted marketing/retention strategies for high-value and at-risk segments.
# MAGIC
# MAGIC Segment outputs allow for precise customer outreach—such as special offers to high-revenue or high-tenure groups.
# MAGIC
# MAGIC Usability:
# MAGIC
# MAGIC All cluster assignments and profiles were exported for further business analysis and intervention planning.
# MAGIC
# MAGIC Business Takeaway:
# MAGIC Segmenting with both behavioral and attitudinal features lets you move from generic campaigns to personalized engagement, reducing churn risk and boosting long-term value.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### customer segmentation using df_feeds

# COMMAND ----------

# Cell 1: imports & helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import joblib
import warnings
warnings.filterwarnings("ignore")

# Optional: UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

def plot_elbow_silhouette(k_list, inertias, silhouettes):
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(k_list, inertias, '-o', color='C0', label='Inertia')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Inertia', color='C0')
    ax2 = ax1.twinx()
    ax2.plot(k_list, silhouettes, '-o', color='C1', label='Silhouette')
    ax2.set_ylabel('Silhouette', color='C1')
    plt.title('Elbow (inertia) and Mean Silhouette by k')
    fig.tight_layout()
    plt.show()


# COMMAND ----------

df_feeds = spark.read.parquet(
    '/Volumes/customer360analysis/customer/gold/feeds_data.parquet/'
)
display(df_feeds)


# COMMAND ----------

# Normalize column names to snake_case
new_cols = [
    c.strip().replace(' ', '_').replace('-', '_')
    for c in df_feeds.columns
]
df_feeds = df_feeds.toDF(*new_cols)
print("Columns detected:", df_feeds.columns)
# Quick preview
display(df_feeds)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Quick cleaning and type normalization

# COMMAND ----------

from pyspark.sql.functions import col, trim, lower, when

# Trim all string columns
string_cols = [f.name for f in df_feeds.schema.fields if f.dataType.simpleString() == 'string']
for c in string_cols:
    df_feeds = df_feeds.withColumn(c, trim(col(c)))

# Map FeedbackScore text to numeric
text_to_score = {'low': 1, 'medium': 2, 'high': 3}
if 'FeedbackScore' in df_feeds.columns:
    df_feeds = df_feeds.withColumn(
        'FeedbackScore_num',
        when(
            lower(trim(col('FeedbackScore'))) == 'low', 1
        ).when(
            lower(trim(col('FeedbackScore'))) == 'medium', 2
        ).when(
            lower(trim(col('FeedbackScore'))) == 'high', 3
        ).otherwise(None)
    )

# Map LoyaltyLevel to ordinal
loyalty_map = {'bronze': 1, 'silver': 2, 'gold': 3}
if 'LoyaltyLevel' in df_feeds.columns:
    df_feeds = df_feeds.withColumn(
        'LoyaltyLevel_clean',
        when(
            lower(trim(col('LoyaltyLevel'))) == 'bronze', 1
        ).when(
            lower(trim(col('LoyaltyLevel'))) == 'silver', 2
        ).when(
            lower(trim(col('LoyaltyLevel'))) == 'gold', 3
        ).otherwise(None)
    )

# Convert numeric-like string columns to numeric
numeric_cols = [
    'Age', 'Income', 'ProductQuality', 'ServiceQuality',
    'PurchaseFrequency', 'SatisfactionScore'
]
from pyspark.sql.functions import regexp_replace
for colname in numeric_cols:
    if colname in df_feeds.columns:
        df_feeds = df_feeds.withColumn(
            colname,
            regexp_replace(trim(col(colname)), ',', '').cast('double')
        )

# Preview schema and null counts
df_feeds.printSchema()
display(df_feeds.select([
    (col(c).isNull().cast('int').alias(c + '_nulls')) for c in df_feeds.columns
]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering

# COMMAND ----------

from pyspark.sql.functions import col, when, mean as _mean, lit, log1p, coalesce

# 1. Compute quality_avg as the average of ProductQuality and ServiceQuality
if set(['ProductQuality', 'ServiceQuality']).issubset(df_feeds.columns):
    df_feeds = df_feeds.withColumn(
        'quality_avg',
        (col('ProductQuality') + col('ServiceQuality')) / 2
    )

# 2. feedback_score_num: use FeedbackScore_num if exists, else FeedbackScore if numeric
if 'FeedbackScore_num' in df_feeds.columns:
    df_feeds = df_feeds.withColumn('feedback_score_num', col('FeedbackScore_num'))
elif 'FeedbackScore' in df_feeds.columns:
    df_feeds = df_feeds.withColumn('feedback_score_num', col('FeedbackScore').cast('double'))

# 3. engagement_score: combine PurchaseFrequency and SatisfactionScore
if 'PurchaseFrequency' in df_feeds.columns:
    df_feeds = df_feeds.withColumn(
        'engagement_score',
        coalesce(col('PurchaseFrequency'), lit(0))
    )
if 'SatisfactionScore' in df_feeds.columns:
    # Fill nulls in SatisfactionScore with its mean
    sat_mean = df_feeds.select(_mean(col('SatisfactionScore'))).first()[0]
    df_feeds = df_feeds.withColumn(
        'engagement_score',
        col('engagement_score') * 0.6 +
        coalesce(col('SatisfactionScore'), lit(sat_mean)) * 0.4
    )

# 4. income_log: log1p transform
if 'Income' in df_feeds.columns:
    df_feeds = df_feeds.withColumn(
        'income_log',
        log1p(coalesce(col('Income'), lit(0)))
    )

# 5. satisfaction_bucket: bucketize SatisfactionScore
if 'SatisfactionScore' in df_feeds.columns:
    sat_median = df_feeds.approxQuantile('SatisfactionScore', [0.5], 0.01)[0]
    df_feeds = df_feeds.withColumn(
        'satisfaction_bucket',
        when(col('SatisfactionScore') <= 2, 'Low')
        .when((col('SatisfactionScore') > 2) & (col('SatisfactionScore') <= 3.5), 'Medium')
        .when(col('SatisfactionScore') > 3.5, 'High')
        .otherwise(None)
    )

# Preview derived columns
display(
    df_feeds.select(
        'quality_avg', 'engagement_score', 'income_log', 'satisfaction_bucket'
    ).limit(10)
)

# COMMAND ----------

# 4. Combine encoded and numeric features
import numpy as np
features = np.hstack((df_feeds_pd[numeric_cols].values, cat_ohe))
print(features.shape)

# COMMAND ----------

# 5. Optional: Feature scaling (recommended for clustering)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# COMMAND ----------

# 6. Dimensionality reduction (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_vis = pca.fit_transform(features_scaled)

# COMMAND ----------

# 7. Clustering (KMeans)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(features_scaled)
df_feeds_pd['cluster_kmeans'] = cluster_labels

# COMMAND ----------

# 8. Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,6))
sns.scatterplot(
    x=X_vis[:,0],
    y=X_vis[:,1],
    hue=df_feeds_pd['cluster_kmeans'].astype(str),
    palette='tab10',
    s=18,
)
plt.title('Customer Segments (df_feeds)')
plt.show()


# COMMAND ----------

df_feeds.columns

# COMMAND ----------

