# Databricks notebook source
# MAGIC %pip install scikit-learn xgboost lightgbm imbalanced-learn mlflow shap joblib

# COMMAND ----------

# Cell 1
import pandas as pd
df = spark.read.parquet(
    '/Volumes/customer360analysis/customer/gold/churn_data.parquet/',
    header = True,
    inferSchema = True
)
display(df)   # your DataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC #### Quick data checks & leakage

# COMMAND ----------

# Basic checks
print(
  f"Rows: {df.count()}, Columns: {len(df.columns)}"
)

display(
  df.groupBy('churn').count()
)

from pyspark.sql.functions import col, sum as spark_sum

display(
  df.select([
    spark_sum(
      col(c).isNull().cast("int")
    ).alias(c)
    for c in df.columns
  ])
)

print(
  df.columns
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2) Define target & drop cols

# COMMAND ----------

# Cell 2
TARGET = 'Churn'           # should be 0/1 or convertible; convert if 'Yes'/'No'
ID_COL = 'Customer ID'     # adjust to your column name

# Drop list (identifiers, leakage)
drop_cols = [ID_COL] + ['churn_score']   # add churn_score if it leaks (decide after inspection)
# Also drop columns with > X% missing or future timestamp features


# COMMAND ----------

# MAGIC %md
# MAGIC #### 3) Train/test (stratified)

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = df.drop(*([TARGET] + drop_cols))
y = df.select(TARGET).toPandas().values.flatten()

X_pd = X.toPandas()

X_train, X_test, y_train, y_test = train_test_split(
    X_pd,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 4) Preprocessing pipeline
# MAGIC
# MAGIC Numeric → StandardScaler (or nothing for tree models but keep in pipeline)
# MAGIC
# MAGIC Categorical → OneHotEncoder(handle_unknown='ignore') for low-cardinality; for high-cardinality consider OrdinalEncoder or target encoding.

# COMMAND ----------

num_types = ["int", "bigint", "double", "float", "decimal", "long", "short"]
num_cols = [
    f.name for f in X.schema.fields
    if any(t in f.dataType.typeName() for t in num_types)
]
cat_cols = [
    f.name for f in X.schema.fields
    if f.dataType.typeName() == "string"
]

print("Numeric:", num_cols)
print("Categorical:", cat_cols)

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
], remainder='drop')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5) Baseline models (Logistic + RandomForest)

# COMMAND ----------

# Cell 6
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(preprocessor, LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42))
pipe_rf = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1))


# COMMAND ----------

# MAGIC %md
# MAGIC #### 6) Cross-validated evaluation function

# COMMAND ----------

# Cell 7
from sklearn.model_selection import StratifiedKFold, cross_validate
scoring = ['roc_auc', 'average_precision', 'precision', 'recall', 'f1']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_report(pipe, X, y, cv=cv):
    res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    for k in res:
        if k.startswith('test_'):
            print(k, res[k].mean(), '+/-', res[k].std())
    return res

print("Logistic CV:")
cv_report(pipe_lr, X_train, y_train)
print("RandomForest CV:")
cv_report(pipe_rf, X_train, y_train)


# COMMAND ----------

# MAGIC %md
# MAGIC #### 7) Handling imbalance: try both class_weight and SMOTE

# COMMAND ----------

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

pipe_rf_smote = Pipeline(steps=[
    ('pre', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

cv_report(pipe_rf_smote, X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 8) Hyperparameter tuning (RandomizedSearchCV) — use XGBoost

# COMMAND ----------

# Cell 9
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)

pipe_xgb = make_pipeline(preprocessor, xgb)

param_dist = {
    "xgbclassifier__n_estimators": [100,200,400],
    "xgbclassifier__max_depth": [3,5,8,12],
    "xgbclassifier__learning_rate": [0.01,0.05,0.1],
    "xgbclassifier__subsample": [0.6,0.8,1.0],
    "xgbclassifier__colsample_bytree": [0.5,0.7,1.0],
    # scale_pos_weight helpful if imbalance ratio known:
    "xgbclassifier__scale_pos_weight": [1, 5, 10, 20]
}

rs = RandomizedSearchCV(
    pipe_xgb, param_dist, n_iter=30, cv=StratifiedKFold(3, shuffle=True, random_state=42),
    scoring='average_precision', n_jobs=-1, verbose=2, random_state=42
)

rs.fit(X_train, y_train)
print("Best params:", rs.best_params_)
print("Best score (AP):", rs.best_score_)
best_model = rs.best_estimator_


# COMMAND ----------

# MAGIC %md
# MAGIC #### 9) Final evaluation on hold-out test
# MAGIC

# COMMAND ----------

# Cell 10
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_proba = best_model.predict_proba(X_test)[:,1]
y_pred = (y_proba >= 0.5).astype(int)   # default threshold; you'll tune it

print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("PR AUC:", average_precision_score(y_test, y_proba))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


# COMMAND ----------

# MAGIC %md
# MAGIC #### 10) Threshold tuning (business tradeoff)

# COMMAND ----------

# Cell 11
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
# choose threshold for desired precision e.g., 0.6
target_precision = 0.6
idx = np.argmax(precision >= target_precision)
thr = thresholds[idx] if idx < len(thresholds) else 0.5
print("Threshold for precision>=%.2f is %.3f (prec=%.2f, rec=%.2f)" % (target_precision, thr, precision[idx], recall[idx]))
# apply thr
y_pred_thr = (y_proba >= thr).astype(int)
print("Precision/Recall/F1 at thr:", precision_score(y_test,y_pred_thr), recall_score(y_test,y_pred_thr), f1_score(y_test,y_pred_thr))


# COMMAND ----------

# MAGIC %md
# MAGIC #### 11) Feature importance & explainability

# COMMAND ----------

# Extract preprocessor feature names
def get_feature_names(preprocessor, num_cols, cat_cols):
    num_names = (
        preprocessor.named_transformers_['num'].get_feature_names_out(num_cols)
        if hasattr(preprocessor.named_transformers_['num'], 'get_feature_names_out')
        else num_cols
    )
    cat_ohe = preprocessor.named_transformers_['cat']
    cat_names = (
        cat_ohe.get_feature_names_out(cat_cols)
        if hasattr(cat_ohe, 'get_feature_names_out')
        else cat_cols
    )
    return list(num_names) + list(cat_names)

# Check available step names in the pipeline
print(best_model.named_steps.keys())

# Use the correct step name, for example 'pre'
feat_names = get_feature_names(
    best_model.named_steps['columntransformer'],
    num_cols,
    cat_cols
)

# COMMAND ----------

# Install SHAP if not already installed
%pip install shap

import shap

# Check available step names in the pipeline
display(best_model.named_steps.keys())

# Replace 'pipeline' and 'columntransformer' with the correct step name for your preprocessor
X_test_pp = best_model.named_steps['columntransformer'].transform(X_test)

# Proceed with SHAP as before
explainer = shap.Explainer(best_model.named_steps['xgbclassifier'])
shap_values = explainer(X_test_pp)
shap.summary_plot(shap_values, X_test_pp)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 12) Save & log model (MLflow)

# COMMAND ----------

# Cell 14
import mlflow, mlflow.sklearn
mlflow.set_experiment("/Shared/Churn_Prediction")

with mlflow.start_run(run_name="xgb_randomized"):
    mlflow.log_params(rs.best_params_)
    mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_proba))
    mlflow.log_metric("test_pr_auc", average_precision_score(y_test, y_proba))
    mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
    mlflow.log_metric("test_recall", recall_score(y_test, y_pred))
    mlflow.sklearn.log_model(best_model, "churn_model")
    # optional: log artifacts (feature importance csv, confusion matrix image)


# COMMAND ----------

# MAGIC %md
# MAGIC #### 13) Productionize predictions

# COMMAND ----------

# Assuming you have a trained model called 'model' and a threshold 'thr'

# Example: select feature columns from df to create full_X
feature_cols = [
    col for col in df.columns
    if col not in ['Customer ID', 'churn_proba', 'churn_pred', 'row_idx']
]
full_X = df.select(feature_cols)

# Convert features Spark DataFrame to Pandas DataFrame
full_X_pd = full_X.toPandas()

# Generate churn probabilities using your model
churn_proba = model.predict_proba(full_X_pd)[:, 1]

# Add a unique row index to Spark DataFrame
df = df.withColumn(
    'row_idx',
    F.monotonically_increasing_id()
)

# Add row index to Pandas DataFrame
full_X_pd['row_idx'] = df.select('row_idx').toPandas()['row_idx']
full_X_pd['churn_proba'] = churn_proba
full_X_pd['churn_pred'] = (full_X_pd['churn_proba'] >= thr).astype(int)

# Convert predictions to Spark DataFrame
preds_spark = spark.createDataFrame(
    full_X_pd[['row_idx', 'churn_proba', 'churn_pred']]
)

# Join predictions back to original DataFrame
df = df.join(
    preds_spark,
    on='row_idx',
    how='inner'
)

# Save CSV
df.select(
    'Customer ID','churn_proba', 'churn_pred'
).toPandas().to_csv('churn_preds.csv', index=False)

# COMMAND ----------

/-89++-*import joblib
joblib.dump(best_model, "churn_model.pkl")

# load later
model_loaded = joblib.load("churn_model.pkl")


# COMMAND ----------

import pandas as pd

# Assume X_test is a pandas DataFrame with the same index as df
df_churn_results = pd.DataFrame({
    "Customer_ID": df.loc[X_test.index, "Customer ID"].values,
    "churn_proba": best_model.predict_proba(X_test)[:, 1],
    "churn_pred": best_model.predict(X_test)
})

df_churn_results.to_csv("churn_predictions.csv", index=False)

# COMMAND ----------

