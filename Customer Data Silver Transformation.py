# Databricks notebook source
# MAGIC %md
# MAGIC Reading data and finding solution to merge multiple datasets to use them to create a unified project that is **Customer 360 Analytics** project.

# COMMAND ----------

# reading csv file of telco customer data
df1 = spark.read.csv('/Volumes/customer360analysis/customer/bronze/WA_Fn-UseC_-Telco-Customer-Churn.csv', header=True, inferSchema=True)
df1.display()

# COMMAND ----------

# reading huggingface telco customer data
df2 = spark.read.csv('/Volumes/customer360analysis/customer/bronze/Huggingface_telco.csv', header=True, inferSchema=True)
df2.display()

# COMMAND ----------

## reading customer satisfaction dataset
df3 = spark.read.csv('/Volumes/customer360analysis/customer/bronze/customer_feedback_satisfaction.csv', header = True, inferSchema=True)
df3.display()

# COMMAND ----------

## reading customer survey data
df4 = spark.read.csv('/Volumes/customer360analysis/customer/bronze/Customer-survey-data.csv', header = True, inferSchema=True)
df4.display()

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see from above datasets, df1(telco customer churn) and df2(huggingface telco) datasets have similar features and can be used for further Exploratory Data Analysis, Feature Engineering, Data Analysis, Customer Segementation, Customer lifetime value, Customer churn, visualization and the df3(feedback and satisfaction data) and df4(survey data) can be used to enrich the visualization and can be feature engineered and integrated to df1 and df2 for better model prediction.
# MAGIC
# MAGIC * Our aim is to find solution to merge df1 and df2 and create datasets for customer segmentation, cltv and churn prediction.
# MAGIC -------
# MAGIC
# MAGIC exploring datasets for merging and feature engineering
# MAGIC
# MAGIC ### EDA on Telco_Customer_Churn Dataset

# COMMAND ----------

# 1. importing libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# COMMAND ----------

df_telco1 = df1

# COMMAND ----------

## 2. reading data
df_telco1.display(5)

# COMMAND ----------

# 3. exploring df_telco1 schema
df_telco1.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see from above schema that the column 'TotalCharges' should be in float or double but it is in string therefore converting it to double (because monthly charges is in double)

# COMMAND ----------

my_ddl_schema = """
    customerID string,
    gender string,
    SeniorCitizen integer,
    Partner string,
    Dependents string,
    tenure integer,
    PhoneService string,
    MultipleLines string,
    InternetService string,
    OnlineSecurity string,
    OnlineBackup string,
    DeviceProtection string,
    TechSupport string,
    StreamingTV string,
    StreamingMovies string,
    Contract string,
    PaperlessBilling string,
    PaymentMethod string,
    MonthlyCharges double,
    TotalCharges double,
    Churn string
"""

# COMMAND ----------

df_telco1 = spark.read.format('csv')\
            .schema(my_ddl_schema)\
            .option('header',True)\
            .load('/Volumes/customer360analysis/customer/bronze/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# COMMAND ----------

# 4. Count Rows & Columns
print("Rows:", df_telco1.count())
print("Columns:", len(df_telco1.columns))


# COMMAND ----------

# 5. Checking missing values
df_telco1.select([sum(col(c).isNull().cast("int")).alias(c) for c in df_telco1.columns]).display()

# COMMAND ----------

# 6. Basic Descriptives (numerical)
numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
df_telco1.select(numerical_cols).describe().display()

# COMMAND ----------

# 7. Unique Values & Distribution (categorical)
categorical_cols = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "Churn"
]

for col_name in categorical_cols:
    print(f"Value counts for {col_name}:")
    df_telco1.groupBy(col_name).count().orderBy(desc("count")).show(truncate=False)


# COMMAND ----------

# 8. Churn Distribution
df_telco1.groupBy("Churn").count().show()

# COMMAND ----------

# 9. Correlation (numerical, where possible)
for num_col in numerical_cols:
    corr = df_telco1.stat.corr(num_col, "Churn")
    print(f"Correlation between {num_col} and Churn: {corr}")

# COMMAND ----------

# 10. Cross-tab Example (Churn vs. Contract)
df_telco1.crosstab("Churn", "Contract").show()

# COMMAND ----------

# 11. Outlier Detection (Optional)
df_telco1.select("MonthlyCharges").summary("min", "25%", "50%", "75%", "max").show()
df_telco1.select("TotalCharges").summary("min", "25%", "50%", "75%", "max").show()

# COMMAND ----------

# 12. Data Types & Cardinality
for c in df_telco1.columns:
    print(f"{c}: {df_telco1.select(c).distinct().count()} unique values")

# COMMAND ----------

# checking for data shape and size
num_rows = df_telco1.count()
num_cols = len(df_telco1.columns)
print(f"Rows: {num_rows}, Columns: {num_cols}")

# COMMAND ----------

## Duplicate rows
dup_count = df_telco1.count() - df_telco1.dropDuplicates().count()
print(f"Duplicate Rows: {dup_count}")

# COMMAND ----------

#  Unique Values (Cardinality)
for c in df_telco1.columns:
    print(f"{c}: {df_telco1.select(approx_count_distinct(c)).first()[0]} unique")


# COMMAND ----------

cat_cols = [
    c
    for c in df_telco1.columns
    if (isinstance(num_cols, list) and c not in num_cols and c != "customerID")
]

for c in cat_cols:
    print(f"\nTop values for {c}:")
    display(
        df_telco1
        .groupBy(c)
        .count()
        .orderBy(desc("count"))
    )

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# COMMAND ----------

# 10. Distribution Plots
# (Spark is not good for plotting directly; use Pandas for small samples)
sample_pd = df_telco1.sample(False, 0.1, seed=42).toPandas()
if 'MonthlyCharges' in sample_pd.columns:
    plt.hist(sample_pd['MonthlyCharges'].dropna(), bins=30)
    plt.title('MonthlyCharges Distribution')
    plt.xlabel('MonthlyCharges')
    plt.ylabel('Frequency')
    plt.show()

# COMMAND ----------

# 11. Outlier Detection (IQR for MonthlyCharges/TotalCharges)
for col_name in ['MonthlyCharges', 'TotalCharges']:
    if col_name in df_telco1.columns:
        quantiles = df_telco1.approxQuantile(col_name, [0.25, 0.5, 0.75], 0.05)
        Q1, Q2, Q3 = quantiles
        IQR = Q3 - Q1
        print(f"{col_name}: Q1={Q1}, Q2={Q2}, Q3={Q3}, IQR={IQR}")


# COMMAND ----------

# 13. Target Variable Analysis
if 'Churn' in df_telco1.columns:
    df_telco1.groupBy("Churn").count().show()
    for c in cat_cols:
        if c != "Churn":
            df_telco1.groupBy("Churn", c).count().orderBy(c).show(5)

# 14. Bivariate Analysis (Numerical vs Target)
if "Churn" in sample_pd.columns:
    import seaborn as sns
    for col_name in ['MonthlyCharges', 'tenure']:
        if col_name in sample_pd.columns:
            sns.boxplot(data=sample_pd, x="Churn", y=col_name)
            plt.title(f"{col_name} by Churn")
            plt.show()

# COMMAND ----------

numerical_cols = [f.name for f in df_telco1.schema.fields if str(f.dataType) in ['IntegerType', 'DoubleType', 'FloatType', 'LongType']]

# Take a random sample (adjust fraction as needed for memory)
pandas_df = df_telco1.select(numerical_cols).sample(False, 0.1, seed=42).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## EDA on Huggingface Telco **Data**

# COMMAND ----------

df_telco2 = df2
df_telco2.display()

# COMMAND ----------

## checking the schema and datatype
df_telco2.printSchema()

# COMMAND ----------

## 2. shape and size of the data
num_rows = df_telco2.count()
num_cols = len(df_telco2.columns)
print(f"Rows: {num_rows}, Columns: {num_cols}")

# COMMAND ----------

from pyspark.sql.functions import col, count, when, isnan

null_counts = df_telco2.select([
    count(
        when(
            col(c).isNull() | 
            (isnan(col(c)) if dict(df_telco2.dtypes)[c] in ["double", "float", "int", "bigint"] else False) | 
            ((col(c) == "") if dict(df_telco2.dtypes)[c] == "string" else False),
            c
        )
    ).alias(c)
    for c in df_telco2.columns
])
display(null_counts)

# COMMAND ----------

# MAGIC %md 
# MAGIC As we can see from above there are many null values 
# MAGIC Churn Category - 3104 and churn Reason - 3104,
# MAGIC Internet Type - 886, offer - 2324
# MAGIC so we need to either fill or drop this null values
# MAGIC
# MAGIC ***If a column has a high % of nulls (e.g., >50%), it’s usually better to drop the column, as imputing may add noise.
# MAGIC If only a few rows are null in a column, you can drop those rows.
# MAGIC If a column is important and has moderate nulls (e.g., 5–30%), consider imputation**.*
# MAGIC
# MAGIC Column	Nulls	% Nulls (approx)	Suggestion
# MAGIC
# MAGIC Churn Category	3104	~31%	Impute ("Unknown") or analyze
# MAGIC
# MAGIC Churn Reason	3104	~31%	Impute ("Unknown") or analyze
# MAGIC
# MAGIC Internet Type	886	~9%	Impute ("Unknown")
# MAGIC
# MAGIC Offer	2324	~23%	Impute ("None") or drop if not useful
# MAGIC
# MAGIC But before we impute the null values with any value we need to find out the reason or pattern for missing value that will help us to decide whether to impute the values with mean/mode/median or with missing/unavailable.

# COMMAND ----------

# Create a flag for missing Churn Reason
df_telco2 = df_telco2.withColumn("ChurnReason_missing", col("Churn Reason").isNull())
# Group by Churn status to see the relationship
df_telco2.groupBy("Churn", "ChurnReason_missing").count().show()

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Sample 500 rows using a fraction, then convert to pandas
sample_size = 500
total_rows = df_telco2.count()
fraction = sample_size / total_rows if total_rows > 0 else 1.0

df_sample = df_telco2.sample(
    withReplacement=False,
    fraction=fraction,
    seed=42
).toPandas()

sns.heatmap(df_sample.isnull(), cbar=False)
plt.show()

# COMMAND ----------

df_telco2.groupBy("Internet Service").agg({"Internet Type":"count"}).show()

# COMMAND ----------

from pyspark.sql.functions import col

# Example: Nulls by Quarter for 'Churn Reason'
df_telco2.withColumn("ChurnReason_missing", col("Churn Reason").isNull()) \
    .groupBy("Quarter", "ChurnReason_missing") \
    .count() \
    .orderBy("Quarter") \
    .show()

# Example: Nulls by State for 'Internet Type'
df_telco2.withColumn("InternetType_missing", col("Internet Type").isNull()) \
    .groupBy("State", "InternetType_missing") \
    .count() \
    .orderBy("State") \
    .show()

# COMMAND ----------

from functools import reduce
from pyspark.sql.functions import col

# Count rows where ANY column is null
null_rows_count = df_telco2.where(
    reduce(
        lambda x, y: x | y,
        [col(c).isNull() for c in df_telco2.columns]
    )
).count()
print(f"Rows with at least one null value: {null_rows_count}")

# Show some random rows with nulls
display(
    df_telco2.where(
        reduce(
            lambda x, y: x | y,
            [col(c).isNull() for c in df_telco2.columns]
        )
    ).limit(10)
)

# COMMAND ----------

# Check if 'Churn Reason' is only null for non-churned customers
df_telco2.groupBy("Churn", col("Churn Reason").isNull().alias("ChurnReason_missing")).count().show()

# Check if 'Internet Type' is only null for customers with no internet service
df_telco2.groupBy("Internet Service", col("Internet Type").isNull().alias("InternetType_missing")).count().show()

# COMMAND ----------

def nulls_by_segment(df, col_name, segment_col):
    df.withColumn(f"{col_name}_missing", col(col_name).isNull()) \
      .groupBy(segment_col, f"{col_name}_missing") \
      .count() \
      .orderBy(segment_col) \
      .show()

# Usage:
nulls_by_segment(df_telco2, "Offer", "Contract")

# COMMAND ----------

# MAGIC %md
# MAGIC After all the analysis for missing values we found the following results:
# MAGIC Column	        Reason for Nulls	          What To Do
# MAGIC
# MAGIC Churn Reason	  Only for churned customers	Impute "Not Churned"/leave null
# MAGIC
# MAGIC Churn Category	Only for churned customers	Impute "Not Churned"/leave null
# MAGIC
# MAGIC Internet Type	  Only for internet customers	Impute "No Internet"/leave null
# MAGIC
# MAGIC Offer	          "No offer" case	            Impute "No Offer" or drop
# MAGIC
# MAGIC Therefore we will impute the values as given above.

# COMMAND ----------

df_telco2 = df_telco2.fillna({
    "Churn Reason": "Not Churned",
    "Churn Category": "Not Churned",
    "Internet Type": "No Internet",
    "Offer": "No Offer"
})

# COMMAND ----------

null_counts = df_telco2.select([
    count(
        when(
            col(c).isNull() | 
            (isnan(col(c)) if dict(df_telco2.dtypes)[c] in ["double", "float", "int", "bigint"] else False) | 
            ((col(c) == "") if dict(df_telco2.dtypes)[c] == "string" else False),
            c
        )
    ).alias(c)
    for c in df_telco2.columns
])
display(null_counts)

# COMMAND ----------

# 6. Duplicate Rows
dup_count = df_telco2.count() - df_telco2.dropDuplicates().count()
print(f"Duplicate Rows: {dup_count}")

# COMMAND ----------

# 7. Unique Values (Cardinality)
for c in df_telco2.columns:
    print(f"{c}: {df_telco2.select(approx_count_distinct(c)).first()[0]} unique")

# COMMAND ----------

df_telco2.display()

# COMMAND ----------

df_telco2.printSchema()

# COMMAND ----------

# 8. Descriptive Stats (Numerical)
num_cols = ["Age",
    "Avg Monthly GB Download",
    "Avg Monthly Long Distance Charges",
    "Churn Score",
    "CLTV",
    "Dependents",
    "Device Protection Plan",
    "Internet Service",
    "Latitude",
    "Longitude",
    "Married",
    "Monthly Charge",
    "Multiple Lines",
    "Number of Dependents",
    "Number of Referrals",
    "Online Backup",
    "Online Security",
    "Paperless Billing",
    "Partner",
    "Phone Service",
    "Population",
    "Premium Tech Support",
    "Referred a Friend",
    "Satisfaction Score",
    "Senior Citizen",
    "Streaming Movies",
    "Streaming Music",
    "Streaming TV",
    "Tenure in Months",
    "Total Charges",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Total Refunds",
    "Total Revenue",
    "Under 30",
    "Unlimited Data",
    "Zip Code",
    "Churn"]
df_telco2.select(num_cols).describe().display()


# COMMAND ----------

# 9. Frequency (Categorical)
cat_cols = [c for c in df_telco2.columns if c not in num_cols and c != "customerID"]
for c in cat_cols:
    print(f"\nTop values for {c}:")
    df_telco2.groupBy(c).count().orderBy(desc("count")).show()


# COMMAND ----------

# 11. Outlier Detection (Optional)
# List of important numerical features
important_features = [
    "Tenure in Months",
    "Monthly Charge",
    "Total Charges",
    "CLTV",
    "Satisfaction Score",
    "Churn Score"
]

# Sample data for plotting (adjust fraction as needed)
pandas_df = df_telco2.select(important_features).sample(False, 0.1, seed=42).toPandas()

# Plot boxplots for only these columns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
pandas_df.boxplot(column=important_features, rot=45, patch_artist=True)
plt.title("Boxplots of Important Features (Outlier Detection)")
plt.tight_layout()
plt.show()


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# 1. Use your important features or all numerical columns
important_features = [
    "Tenure in Months",
    "Monthly Charge",
    "Total Charges",
    "CLTV",
    "Satisfaction Score",
    "Churn Score"
]

# 2. Sample Spark DataFrame and convert to Pandas
pandas_df = df_telco2.select(important_features).sample(False, 0.1, seed=42).toPandas()

# 3. Compute correlation matrix (Pandas)
corr_matrix = pandas_df.corr()

# 4. Plot correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix Heatmap of Important Features")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA on Dataset 3 (df3) that is customer feedback and satisfaction

# COMMAND ----------

df_feeds = df3
df_feeds.display()

# COMMAND ----------

df_feeds.printSchema()

# COMMAND ----------

## 2. shape and size of the data
num_rows = df_feeds.count()
num_cols = len(df_feeds.columns)
print(f"Rows: {num_rows}, Columns: {num_cols}")


# COMMAND ----------

from pyspark.sql.functions import col, count, when, isnan

null_counts = df_feeds.select([
    count(
        when(
            col(c).isNull() | 
            (isnan(col(c)) if dict(df_feeds.dtypes)[c] in ["double", "float", "int", "bigint"] else False) | 
            ((col(c) == "") if dict(df_feeds.dtypes)[c] == "string" else False),
            c
        )
    ).alias(c)
    for c in df_feeds.columns
])
display(null_counts)

# COMMAND ----------

# Duplicate Rows
dup_count = df_feeds.count() - df_feeds.dropDuplicates().count()
print(f"Duplicate Rows: {dup_count}")

# COMMAND ----------

# 7. Unique Values (Cardinality)
for c in df_feeds.columns:
    print(f"{c}: {df_feeds.select(approx_count_distinct(c)).first()[0]} unique")

# COMMAND ----------

df_feeds.printSchema()

# COMMAND ----------

# 8. Descriptive Stats (Numerical)
nums_cols = ["Age",
    "Income",
    "ProductQuality",
    "ServiceQuality",
    "PurchaseFrequency",
    "SatisfactionScore"]
df_feeds.select(nums_cols).describe().display()

# COMMAND ----------

# Distribution plots (sample to pandas)
if nums_cols:
    sample_pd = df_feeds.select(nums_cols).sample(False, 0.1, seed=42).toPandas()
    for col_name in nums_cols:
        if col_name in sample_pd.columns:
            plt.hist(sample_pd[col_name].dropna(), bins=30)
            plt.title(f'{col_name} Distribution')
            plt.xlabel(col_name)
            plt.ylabel('Frequency')
            plt.show()

# COMMAND ----------

df_feeds.printSchema()

# COMMAND ----------

# 2. Categorical columns: value counts and unique values
cat_cols = ['Gender','Country','FeedbackScore','LoyaltyLevel']

for col_name in cat_cols:
    print(f"Value counts for {col_name}:")
    df_feeds.groupBy(col_name).count().orderBy('count', ascending=False).show(5)
    uniq_count = df_feeds.select(col_name).distinct().count()
    print(f"Unique values in {col_name}: {uniq_count}")

# COMMAND ----------

# 3. Potential join keys: look for customerID or other unique identifiers
possible_keys = [c for c in df_feeds.columns if 'id' in c.lower() or 'customer' in c.lower()]
print(f"Potential join keys: {possible_keys}")
for key in possible_keys:
    uniq = df_feeds.select(key).distinct().count()
    print(f"Unique values in {key}: {uniq} (should match row count for a perfect key)")


# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA on df4 that is survey data

# COMMAND ----------

df_survey = df4
df_survey.display()

# COMMAND ----------

df_survey.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC as the name of attribute is too long therefore we will shorten it. 

# COMMAND ----------

from pyspark.sql.functions import col

old_names = df_survey.columns
new_names = [
    "customer",
    "delivery_satisfaction",
    "food_quality_satisfaction",
    "delivery_speed_satisfaction",
    "order_accuracy"
    # add more if needed
]
df_survey = df_survey.toDF(*new_names)
print(df_survey.columns)

# COMMAND ----------

from pyspark.sql.functions import col, count, when, isnan

null_counts = df_survey.select([
    count(
        when(
            col(c).isNull() | 
            (
                isnan(col(c)) 
                if dict(df_survey.dtypes)[c] in ["double", "float", "int", "bigint"] 
                else False
            ) | 
            (
                (col(c) == "") 
                if dict(df_survey.dtypes)[c] == "string" 
                else False
            ),
            c
        )
    ).alias(f"{c}")
    for c in df_survey.columns
])
display(null_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC There are alot of null values 
# MAGIC delivery_satisfaction = 418,
# MAGIC food_quality_satisfaction = 252,
# MAGIC delivery_speed_satisfaction = 239, 
# MAGIC order accuracy = 660

# COMMAND ----------

df_survey = df_survey.dropna()

# COMMAND ----------



# COMMAND ----------

# Duplicate Rows
dup_count = df_survey.count() - df_survey.dropDuplicates().count()
print(f"Duplicate Rows: {dup_count}")

# COMMAND ----------

# 7. Unique Values (Cardinality)
for c in df_survey.columns:
    print(f"{c}: {df_survey.select(approx_count_distinct(c)).first()[0]} unique")

# COMMAND ----------

df_survey.printSchema()

# COMMAND ----------

# 8. Descriptive Stats (Numerical)
nums_cols = [
    "delivery_satisfaction",
    "food_quality_satisfaction",
    "delivery_speed_satisfaction"]
df_survey.select(nums_cols).describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC reviewing all datasets before writing them to silver layer
# MAGIC

# COMMAND ----------

df_telco1.display()

# COMMAND ----------

df_telco2.display()

# COMMAND ----------

df_feeds.display()

# COMMAND ----------

df_survey.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Writing files to silver layer in parquet format 

# COMMAND ----------

df_telco1.write.mode("overwrite").parquet("/Volumes/customer360analysis/customer/silver/telco_churn_silver.parquet")
df_telco2.write.mode("overwrite").parquet("/Volumes/customer360analysis/customer/silver/huggingface_telco_silver.parquet")
df_feeds.write.mode("overwrite").parquet("/Volumes/customer360analysis/customer/silver/feedback_satisfaction_silver.parquet")
df_survey.write.mode("overwrite").parquet("/Volumes/customer360analysis/customer/silver/customer_survey_silver.parquet")

# COMMAND ----------

# Replace <container>, <account> with your actual values
silver_path = "abfss://silver@rgcustomer.dfs.core.windows.net/customer360/silver/dataset/customer_survey_silver.parquet"
spark.conf.set("fs.azure.account.key.rgcustomer.dfs.core.windows.net", "WM623IbaRiNzMIZpS3aq00qgiVHG7Y23m3sOqkiwoAr4oKUS35RIzM0xuuW7H5zxCzNlpwFEeLS+AStkvXtPQ==")
df_survey.write.mode("overwrite").parquet(silver_path)

# COMMAND ----------

