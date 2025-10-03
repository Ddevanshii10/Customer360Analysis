# Databricks notebook source
df_cltv = spark.read.parquet(
    '/Volumes/customer360analysis/customer/gold/cltv_data.parquet/',
    header = True,
    infer_schema = True
)
df_cltv.display()

# COMMAND ----------

df_segmentation = spark.read.csv(
    '/Volumes/customer360analysis/customer/gold/segmentation_result.csv',
    header=True,
    inferSchema=True
)

display(df_segmentation)

# COMMAND ----------

# MAGIC %md
# MAGIC Quick Churn Prediction to get churn probability 

# COMMAND ----------

# Define your DataFrame and target column
df = df_cltv  # Your Spark DataFrame
target_col = 'Churn'

# Drop identifier columns if present
drop_cols = [col for col in [target_col, 'customer_id'] if col in df.columns]
X = df.drop(*drop_cols)
y = df.select(target_col)

# Convert to pandas for scikit-learn
X_pd = X.toPandas()
y_pd = y.toPandas().squeeze()

# Identify numeric and categorical columns
num_cols = X_pd.select_dtypes(include=['number']).columns.tolist()
cat_cols = X_pd.select_dtypes(exclude=['number']).columns.tolist()

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

# Full pipeline with classifier
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pd,
    y_pd,
    test_size=0.2,
    random_state=42
)

# Fit pipeline
pipe.fit(X_train, y_train)

# Predict probabilities
churn_prob = pipe.predict_proba(X_test)[:, 1]

result = X_test.copy()
result['churn_probability'] = churn_prob
display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC merging cltv dataset with segmentation so that we get good results for cltv prediction

# COMMAND ----------

df_merged = df_cltv.join(df_cltv, on=[col for col in df_cltv.columns if col in df_cltv.columns], how='inner')
display(df_merged)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we have merged the datasets we need to perfrom feature engineering for cltv prediction and before lets have quick check of data quality and eda along with some important visualizations

# COMMAND ----------

import pyspark.sql.functions as F

# Data Quality Checks
print("CLTV DataFrame Info:")
df_cltv.printSchema()
print("CLTV Null Counts:")
display(
    df_cltv.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df_cltv.columns
    ])
)

print("Segmentation DataFrame Info:")
df_segmentation.printSchema()
print("Segmentation Null Counts:")
display(
    df_segmentation.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df_segmentation.columns
    ])
)

# Basic EDA: Summary statistics
print("CLTV Summary Statistics:")
display(df_cltv.describe())
print("Segmentation Summary Statistics:")
display(df_segmentation.describe())

# Value counts for categorical columns (top 3)
cat_cols_cltv = [
    f.name for f in df_cltv.schema.fields
    if str(f.dataType) == 'StringType'
]
for col_name in cat_cols_cltv[:3]:
    print(f"Value counts for {col_name}:")
    display(
        df_cltv.groupBy(col_name).count().orderBy('count', ascending=False)
    )

cat_cols_seg = [
    f.name for f in df_segmentation.schema.fields
    if str(f.dataType) == 'StringType'
]
for col_name in cat_cols_seg[:3]:
    print(f"Value counts for {col_name}:")
    display(
        df_segmentation.groupBy(col_name).count().orderBy('count', ascending=False)
    )

# Visualizations
import matplotlib.pyplot as plt

# Histogram for a numeric column in CLTV
num_cols_cltv = [
    f.name for f in df_cltv.schema.fields
    if str(f.dataType) in ['DoubleType', 'IntegerType', 'LongType', 'FloatType']
]
if num_cols_cltv:
    sample_pd = (
        df_cltv.select(num_cols_cltv[0])
        .dropna()
        .sample(fraction=0.1, seed=42)
        .toPandas()
    )
    plt.figure(figsize=(8, 4))
    plt.hist(sample_pd[num_cols_cltv[0]], bins=30, color='skyblue')
    plt.title(f'Histogram of {num_cols_cltv[0]}')
    plt.xlabel(num_cols_cltv[0])
    plt.ylabel('Frequency')
    plt.show()

# Bar plot for a categorical column in Segmentation
if cat_cols_seg:
    cat_pd = (
        df_segmentation.groupBy(cat_cols_seg[0])
        .count()
        .orderBy('count', ascending=False)
        .limit(10)
        .toPandas()
    )
    plt.figure(figsize=(8, 4))
    plt.bar(cat_pd[cat_cols_seg[0]], cat_pd['count'], color='orange')
    plt.title(f'Bar Plot of {cat_cols_seg[0]}')
    plt.xlabel(cat_cols_seg[0])
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# COMMAND ----------

TARGET = "CLTV"
ID_COL = "Customer_ID"

# Drop ID + target
features = [c for c in df_cltv.columns if c not in [ID_COL, TARGET]]

X = df_cltv[features]
y = df_cltv[TARGET]


# COMMAND ----------

numeric_features = [
    "Age","Tenure_in_Months","Monthly_Charge","Total_Charges","Total_Revenue",
    "avg_revenue_per_month","total_spend_ratio","num_services",
    "monthly_charge_x_tenure","log_total_revenue","Satisfaction_Score","churn_proba","dist_to_centroid"
]
numeric_features = [c for c in numeric_features if c in X.columns]

categorical_features = [c for c in ["Gender","Contract","Payment_Method","Offer","Unlimited_Data","Paperless_Billing","cluster_kmeans"] if c in X.columns]


# COMMAND ----------

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, numeric_features),
    ("cat", cat_transformer, categorical_features)
])


# COMMAND ----------

# Split the PySpark DataFrame into train and test sets
train_df, test_df = X.randomSplit(
    [0.8, 0.2],
    seed=42
)

display(train_df)
display(test_df)

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

rf = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42))
])

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")


# COMMAND ----------

# MAGIC %pip install xgboost lightgbm

# COMMAND ----------

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import loguniform
import numpy as np

# Optionally apply log1p transform to target if CLTV is skewed
y_train_trans = np.log1p(y_train)
y_test_trans = np.log1p(y_test)

# XGBoost hyperparameter search
xgb = Pipeline([
    ("preprocess", preprocessor),
    ("model", XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1))
])
xgb_param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": loguniform(0.01, 0.2),
    "model__subsample": [0.7, 0.9, 1.0]
}
xgb_search = RandomizedSearchCV(
    xgb, xgb_param_grid, n_iter=10, cv=3, scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1
)
xgb_search.fit(X_train, y_train_trans)
y_pred_xgb = np.expm1(xgb_search.predict(X_test))
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"XGBoost RMSE: {rmse_xgb:.2f}")

# LightGBM hyperparameter search
lgbm = Pipeline([
    ("preprocess", preprocessor),
    ("model", LGBMRegressor(objective="regression", random_state=42, n_jobs=-1))
])
lgbm_param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7, -1],
    "model__learning_rate": loguniform(0.01, 0.2),
    "model__subsample": [0.7, 0.9, 1.0]
}
lgbm_search = RandomizedSearchCV(
    lgbm, lgbm_param_grid, n_iter=10, cv=3, scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1
)
lgbm_search.fit(X_train, y_train_trans)
y_pred_lgbm = np.expm1(lgbm_search.predict(X_test))
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
print(f"LightGBM RMSE: {rmse_lgbm:.2f}")

# COMMAND ----------

from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

scoring = {
    "MAE": make_scorer(mean_absolute_error),
    "RMSE": make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred))),
    "R2": make_scorer(r2_score),
    "MAPE": make_scorer(mape, greater_is_better=False)
}

models = {
    "RandomForest": Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42))
    ]),
    "XGBoost": Pipeline([
        ("preprocess", preprocessor),
        ("model", XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1))
    ]),
    "LightGBM": Pipeline([
        ("preprocess", preprocessor),
        ("model", LGBMRegressor(objective="regression", random_state=42, n_jobs=-1))
    ])
}

results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    cv_result = cross_validate(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring=scoring,
        return_train_score=False
    )
    results.append({
        "Model": name,
        "MAE": np.mean(cv_result["test_MAE"]),
        "RMSE": np.mean(cv_result["test_RMSE"]),
        "R2": np.mean(cv_result["test_R2"]),
        "MAPE": -np.mean(cv_result["test_MAPE"])
    })

import pandas as pd
cv_summary = pd.DataFrame(results)
display(cv_summary)

# COMMAND ----------

import joblib
joblib.dump(rf, "cltv_model.pkl")


# COMMAND ----------

