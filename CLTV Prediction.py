# Databricks notebook source
df = spark.read.parquet(
    '/Volumes/customer360analysis/customer/gold/cltv_data.parquet/'
)
display(df)

# COMMAND ----------

df.columns

# COMMAND ----------

#importing necessary libraries
# Cell 0: imports & basic setup
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from scipy.stats import randint, uniform

import joblib


# COMMAND ----------

# Define y as the target column - CLTV 
y = df.select('CLTV')

# Define X by dropping the target column CLTV and Customer_ID as it is of no use
X = df.drop('Customer_ID', 'CLTV')

# COMMAND ----------

print(f"X shape: ({X.count()}, {len(X.columns)})")
print(f"y shape: ({y.count()}, {len(y.columns)})")

# COMMAND ----------

X.printSchema()

# COMMAND ----------

y.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Identifying feature types - categorical and numerical

# COMMAND ----------

X.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Train/test split

# COMMAND ----------

X_pd = X.toPandas()
y_pd = y.toPandas()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_pd,
    y_pd,
    test_size=0.2,
    random_state=42
)

# COMMAND ----------

# MAGIC %md
# MAGIC Categorical and Numerical featues

# COMMAND ----------

cat_features = [
    'Gender','Contract','Offer','Payment_Method','tenure_bucket'
]

# COMMAND ----------

num_features = [col for col in X.columns if col not in cat_features]

# COMMAND ----------

# MAGIC %md
# MAGIC ###### processing features , standard scaler num features and OneHotEncoding cat features

# COMMAND ----------

# Cell 2: build preprocessor and pipelines
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ],
    remainder='drop'  # drop any other cols
)

# Pipelines
pipeline_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

pipeline_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Cross Validation

# COMMAND ----------

# Cell 3: cross-validate RandomForest on training data
scoring = {
    'r2': 'r2',
    'neg_mse': 'neg_mean_squared_error',
    'neg_mae': 'neg_mean_absolute_error'
}

cv_results = cross_validate(pipeline_rf, X_train, y_train, cv=5, scoring=scoring, return_train_score=False, n_jobs=-1)
# convert neg metrics
rmse_scores = np.sqrt(-cv_results['test_neg_mse'])
mae_scores = -cv_results['test_neg_mae']
r2_scores = cv_results['test_r2']

print("CV RMSE (5-fold):", rmse_scores.mean(), "±", rmse_scores.std())
print("CV MAE (5-fold):", mae_scores.mean(), "±", mae_scores.std())
print("CV R2  (5-fold):", r2_scores.mean(), "±", r2_scores.std())


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Hyperparameter tuning

# COMMAND ----------

# Cell 4: RandomizedSearchCV on pipeline_rf (on X_train)
param_dist = {
    "model__n_estimators": randint(50, 400),
    "model__max_depth": randint(3, 30),
    "model__min_samples_split": randint(2, 10),
    "model__min_samples_leaf": randint(1, 8),
    "model__max_features": ["auto", "sqrt", "log2", 0.5, 0.7],
    "model__bootstrap": [True, False]
}

rs = RandomizedSearchCV(
    estimator=pipeline_rf,
    param_distributions=param_dist,
    n_iter=30,               # reduce if compute limited
    cv=3,                    # use 3-fold for speed (increase to 5 if you can)
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rs.fit(X_train, y_train)
print("Best params:", rs.best_params_)
print("Best CV RMSE:", np.sqrt(-rs.best_score_))


# COMMAND ----------

# Extract tuned parameters (drop the "model__" prefix)
best_params_clean = {k.replace("model__", ""): v for k,v in rs.best_params_.items() if k.startswith("model__")}

# Rebuild the pipeline with tuned params
final_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(**best_params_clean, random_state=42, n_jobs=-1))
])

# Fit on training data
final_rf.fit(X_train, y_train)


# COMMAND ----------

# Predict on held-out test set
y_pred = final_rf.predict(X_test)
y_pred


# COMMAND ----------

# MAGIC %md
# MAGIC ###### Evaluate best models

# COMMAND ----------

# Cell 5: evaluate best estimator on X_test


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Test MAE:", mae)
print("Test RMSE:", rmse)
print("Test R2:", r2)



# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Feature Importance

# COMMAND ----------

# Cell 6: helper to get feature names from preprocessor
def get_feature_names_from_column_transformer(column_transformer):
    feature_names = []
    # for sklearn >= 1.0
    for name, transformer, cols in column_transformer.transformers_:
        if name == 'remainder' and transformer == 'drop':
            continue
        if transformer == 'passthrough':
            feature_names.extend(cols)
        else:
            # transformer may be a pipeline or directly OneHotEncoder/Scaler
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    # For OneHotEncoder and others that accept input feature names
                    names = list(transformer.get_feature_names_out(cols))
                except Exception:
                    # Sometimes get_feature_names_out expects no args
                    names = list(transformer.get_feature_names_out())
                feature_names.extend(names)
            else:
                # fallback: use original col names
                if isinstance(cols, (list, tuple, np.ndarray)):
                    feature_names.extend(list(cols))
                else:
                    feature_names.append(cols)
    return feature_names

# Extract preprocessor & fitted model
preproc = best_rf.named_steps['preprocessor']
rf_model = best_rf.named_steps['model']

# transform preprocessor to get feature names (preproc must be fitted)
feature_names = get_feature_names_from_column_transformer(preproc)
len(feature_names), feature_names[:20]

# get importances
importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
feat_imp = feat_imp.sort_values("importance", ascending=False).reset_index(drop=True)
print(feat_imp.head(30))

# Optional: plot top 20
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.barh(feat_imp['feature'].head(20)[::-1], feat_imp['importance'].head(20)[::-1])
plt.xlabel("Feature Importance")
plt.title("Top 20 Feature Importances (RandomForest)")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Feature Selection and retrain

# COMMAND ----------

# Cell 7: select top-k features (e.g., top 30)
top_k = 20
top_features = feat_imp['feature'].head(top_k).tolist()

# Transform train/test with preprocessor, convert to DataFrame with col names
X_train_trans = preproc.transform(X_train)
X_test_trans = preproc.transform(X_test)

X_train_df = pd.DataFrame(X_train_trans, columns=feature_names)
X_test_df  = pd.DataFrame(X_test_trans,  columns=feature_names)

X_train_top = X_train_df[top_features]
X_test_top  = X_test_df[top_features]

# Fit a new RandomForest on top features
rf2 = RandomForestRegressor(**{k.replace("model__", ""): v for k,v in rs.best_params_.items() if k.startswith("model__")})
rf2.fit(X_train_top, y_train)

# Evaluate
y_pred2 = rf2.predict(X_test_top)
print("Test RMSE (top-k):", np.sqrt(mean_squared_error(y_test, y_pred2)))
print("Test R2  (top-k):", r2_score(y_test, y_pred2))


# COMMAND ----------

# MAGIC %md
# MAGIC ##### MLflow tracking

# COMMAND ----------

import mlflow
import mlflow.sklearn

# Set experiment with full path (will create if not exists)
mlflow.set_experiment(
    "/Users/joshidevanshi1012@gmail.com/CLTV_regression"
)

with mlflow.start_run(run_name="rf_randomized_search_best"):
    mlflow.sklearn.log_model(
    sk_model=best_rf,
    artifact_path="rf_model",
    input_example=X_test[:5])
    mlflow.log_params({k: v for k, v in rs.best_params_.items()})
    mlflow.log_metric("cv_rmse", np.sqrt(-rs.best_score_))
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_r2", r2)
    mlflow.sklearn.log_model(best_rf, artifact_path="rf_model")
    feat_imp.to_csv("/tmp/feature_importance.csv", index=False)
    mlflow.log_artifact("/tmp/feature_importance.csv", artifact_path="artifacts")
    print("Logged to MLflow run:", mlflow.active_run().info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Evaluate Models

# COMMAND ----------

f

# COMMAND ----------

import matplotlib.pyplot as plt
plt.hist(y_pd, bins=50)
plt.title("Distribution of CLTV")
plt.show()


# COMMAND ----------

import seaborn as sns
corr = X_pd.corrwith(y_pd['CLTV'])
sns.barplot(x=corr.values, y=corr.index)
plt.title("Feature Correlation with CLTV")
plt.show()


# COMMAND ----------

