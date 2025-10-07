# Databricks notebook source
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC This is the gold layer transformation where we will create final, analytics ready datasets for each business problem (churn prediction, customer segmentation and cltv) and apply feature engineering, standardization, and normalization. 

# COMMAND ----------

df_telco1 = (spark.read.format('parquet')
            .option('inferSchema',True)
            .option('header',True)
            .load('/Volumes/customer360analysis/customer/silver/telco_churn_silver.parquet'))
df_telco1.printSchema()

# COMMAND ----------

num_rows = df_telco1.count()
num_cols = len(df_telco1.columns)
print(f"Rows: {num_rows}, Columns: {num_cols}")

# COMMAND ----------

df_telco2 = (spark.read.format('parquet')
            .option('inferSchema',True)
            .option('header',True)
            .load('/Volumes/customer360analysis/customer/silver/huggingface_telco_silver.parquet'))
df_telco2.printSchema()

# COMMAND ----------

num_rows = df_telco2.count()
num_cols = len(df_telco2.columns)
print(f"Rows: {num_rows}, Columns: {num_cols}")

# COMMAND ----------

df_feeds = (spark.read.format('parquet')
            .option('inferSchema',True)
            .option('header',True)
            .load('/Volumes/customer360analysis/customer/silver/feedback_satisfaction_silver.parquet'))
df_feeds.printSchema()

# COMMAND ----------

num_rows = df_feeds.count()
num_cols = len(df_feeds.columns)
print(f"Rows: {num_rows}, Columns: {num_cols}")

# COMMAND ----------

# MAGIC %md 
# MAGIC For Customer LifeTime Value we will extract some relevant features from telco2 dataset and than we will perform feature engineering on that dataset in this layer.
# MAGIC
# MAGIC The relevant features are : 
# MAGIC * demographics - age, gender, married, dependents
# MAGIC * subscription/contract details - contract(monthly, yearly), tenure in months, offer, payment method
# MAGIC * usage and speed - monthly charge, total charge, avg, monthly gb download, total long dist charges, total rvenue
# MAGIC * service feature - internet service, streaming service, tech support, device protection
# MAGIC * target variable - cltv
# MAGIC
# MAGIC and some optional features as well.

# COMMAND ----------

selected_cols = [
    "Customer ID", "Age", "Gender", "Married", "Dependents",  # or "Number of Dependents"
    "Contract", "Tenure in Months", "Offer", "Payment Method",
    "Monthly Charge", "Total Charges", "Avg Monthly GB Download",
    "Total Long Distance Charges", "Total Revenue",
    "Internet Service", "Streaming TV", "Streaming Movies", "Streaming Music",
    "Premium Tech Support", "Device Protection Plan", "Online Security", "Online Backup",
    "Referred a Friend", "Number of Referrals", "Satisfaction Score", 
    "Unlimited Data", "Paperless Billing", "CLTV", "Churn","Churn Reason","Churn Score"
]

# Drop columns that don't exist in your data or that you decide not to use
final_cols = [col for col in selected_cols if col in df_telco2.columns]
df_cltv = df_telco2.select(final_cols)

# COMMAND ----------

# reading df_cltv data
df_cltv.display()

# COMMAND ----------

# checking Schema
df_cltv.printSchema()

# COMMAND ----------

# cross-checking once for null values
from pyspark.sql.functions import *
df_cltv.select([sum(col(c).isNull().cast("int")).alias(c) for c in df_cltv.columns]).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC 1. Creating some additional features

# COMMAND ----------

# creating a new column tenure bucket for more better prediction
df_cltv = df_cltv.withColumn(
            'tenure_bucket',
            when(col('Tenure in Months')<12, '<1yr')
            .when((col('Tenure in Months')>=12) & (col('Tenure in Months')<24), '1-2yr')
            .otherwise('2+yrs')
)

df_cltv.display()

# COMMAND ----------

# Average revenue per month
df_cltv = df_cltv.withColumn(
    "avg_revenue_per_month",
    col("Total Revenue") / (col("Tenure in Months") + 1)
)

# Total spend ratio
df_cltv = df_cltv.withColumn(
    "total_spend_ratio",
    col("Total Charges") / (col("Total Revenue") + 1)
)

# Is high spender
df_cltv = df_cltv.withColumn(
    "is_high_spender",
    when(col("Monthly Charge") > 80, 1).otherwise(0)
)


# COMMAND ----------

df_cltv.display()

# COMMAND ----------

from functools import reduce
from pyspark.sql.functions import col, when, log1p

service_cols = [
    "Streaming TV", "Streaming Movies", "Streaming Music",
    "Premium Tech Support", "Device Protection Plan",
    "Online Security", "Online Backup"
]

df_cltv = df_cltv.withColumn(
    "num_services",
    reduce(
        lambda a, b: a + b,
        [when(col(c) == 1, 1).otherwise(0) for c in service_cols]
    )
)

df_cltv = df_cltv.withColumn(
    "monthly_charge_x_tenure",
    col("Monthly Charge") * col("Tenure in Months")
)

df_cltv = df_cltv.withColumn(
    "log_total_revenue",
    log1p(col("Total Revenue"))
)

# COMMAND ----------

df_cltv.display()

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Encoding Categorical Columns

# COMMAND ----------

# Remove spaces from existing column names
new_columns = [col.replace(" ", "_") for col in df_cltv.columns]

# Rename columns and assign back to the dataframe
df_cltv = df_cltv.toDF(*new_columns)


# COMMAND ----------

df_cltv.columns

# COMMAND ----------

# MAGIC %md 
# MAGIC Writing df_cltv to golden layer folder as it is ready for training a model

# COMMAND ----------

df_cltv.write.mode("overwrite").parquet("/Volumes/customer360analysis/customer/gold/cltv_data.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating Dataset for churn prediction

# COMMAND ----------

df_telco2.columns

# COMMAND ----------

churn_features = [
    "Customer ID",
    "Age",
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure in Months",
    "Contract",
    "Payment Method",
    "Paperless Billing",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Device Protection Plan",
    "Monthly Charge",
    "Total Charges",
    "Churn"
    # Plus engineered features: tenure_bucket, num_services_subscribed, is_high_spender
]

# Drop columns that don't exist in your data or that you decide not to use
req_cols = [col for col in churn_features if col in df_telco2.columns]
df_churn= df_telco2.select(req_cols)
df_churn.columns

# COMMAND ----------

df_churn.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Performing Feature Engineering by adding some engineered columns

# COMMAND ----------

from pyspark.sql.functions import when, col

df_churn = df_churn.withColumn(
    'tenure_bucket',
    when(col('Tenure in Months') < 12, '<1yr')
    .when((col('Tenure in Months') >= 12) & (col('Tenure in Months') < 24), '1-2yr')
    .otherwise('2+yrs')
)
display(df_churn)

# COMMAND ----------

from functools import reduce
from pyspark.sql.functions import col, when

service_cols = [
    "Phone Service", "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Premium Tech Support", "Streaming TV", "Streaming Movies", "Device Protection Plan"
]

df_churn = df_churn.withColumn(
    "num_services_subscribed",
    reduce(
        lambda a, b: a + b,
        [
            when(
                col(c).cast("string").isin("Yes", "1", "Fiber optic", "DSL"),
                1
            ).otherwise(0)
            for c in service_cols
        ]
    )
)

display(df_churn)

# COMMAND ----------

# MAGIC %md
# MAGIC cross checking for null values

# COMMAND ----------

from pyspark.sql.functions import col, sum, when

display(
    df_churn.select([
        sum(
            when(col(c).isNull(), 1).otherwise(0)
        ).alias(c)
        for c in df_churn.columns
    ])
)

# COMMAND ----------

# MAGIC %md
# MAGIC Writing df_churn to gold layer

# COMMAND ----------

df_churn.write.mode("overwrite").parquet("/Volumes/customer360analysis/customer/gold/churn_data.parquet")

# COMMAND ----------

df_telco2.columns

# COMMAND ----------

df_feeds.columns

# COMMAND ----------

# MAGIC %md
# MAGIC For creating dataset for customer segmentation we will join telco2 and feeds data to get richer dataset but the customer ids are different so we cant add these two datasets.
# MAGIC
# MAGIC But now we will create cust_seg data from telco2 datasets and by merging some features of churn and cltv datasets

# COMMAND ----------

from pyspark.sql import functions as F

# Start with segmentation base
df_segmentation = df_telco2.select(
    "Customer ID", "Age", "Gender", "Married", "Dependents",
    "Contract", "Tenure in Months", "Offer", "Payment Method",
    "Monthly Charge", "Total Charges", "Avg Monthly GB Download",
    "Total Long Distance Charges", "Total Revenue", "Internet Service",
    "Streaming TV", "Streaming Movies", "Streaming Music",
    "Premium Tech Support", "Device Protection Plan",
    "Online Security", "Online Backup",
    "Referred a Friend", "Number of Referrals",
    "Satisfaction Score", "Unlimited Data", "Paperless Billing"
)

# Create derived features directly in segmentation
df_segmentation = (
    df_segmentation
    # bucket tenure
    .withColumn(
        "tenure_bucket",
        F.when(F.col("Tenure in Months") < 12, "0-12 Months")
         .when((F.col("Tenure in Months") >= 12) & (F.col("Tenure in Months") < 24), "12-24 Months")
         .when((F.col("Tenure in Months") >= 24) & (F.col("Tenure in Months") < 48), "24-48 Months")
         .otherwise("48+ Months")
    )
    # revenue per month
    .withColumn("avg_revenue_per_month",
                F.col("Total Revenue") / F.when(F.col("Tenure in Months") > 0, F.col("Tenure in Months")).otherwise(1))
    # total spend ratio
    .withColumn("total_spend_ratio",
                F.col("Total Charges") / F.when(F.col("Total Revenue") > 0, F.col("Total Revenue")).otherwise(1))
    # high spender flag
    .withColumn("is_high_spender",
                F.when(F.col("Monthly Charge") > 80, 1).otherwise(0))
    # number of subscribed services
    .withColumn("num_services",
                F.col("Streaming TV") + F.col("Streaming Movies") +
                F.col("Streaming Music") + F.col("Premium Tech Support") +
                F.col("Device Protection Plan") + F.col("Online Security") +
                F.col("Online Backup") + F.col("Internet Service"))
    # interaction features
    .withColumn("monthly_charge_x_tenure",
                F.col("Monthly Charge") * F.col("Tenure in Months"))
    .withColumn("log_total_revenue",
                F.log1p(F.col("Total Revenue")))
)

# Clean column names for ML
for col in df_segmentation.columns:
    df_segmentation = df_segmentation.withColumnRenamed(col, col.replace(" ", "_"))

# Inspect
print("Segmentation dataset rows:", df_segmentation.count())
print("Segmentation dataset cols:", len(df_segmentation.columns))
df_segmentation.show(5, truncate=False)


# COMMAND ----------

df_segmentation.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Cross verify the segment dataset for nulls

# COMMAND ----------

from pyspark.sql.functions import col, sum, when

display(
    df_segmentation.select([
        sum(
            when(col(c).isNull(), 1).otherwise(0)
        ).alias(c)
        for c in df_segmentation.columns
    ])
)

# COMMAND ----------

# MAGIC %md
# MAGIC Writing both df_segmentation and df_feeds data to gold layer as we will use them for customer segmentation - clustering

# COMMAND ----------

df_segmentation.write.mode("overwrite").parquet("/Volumes/customer360analysis/customer/gold/segment_data.parquet")
df_feeds.write.mode("overwrite").parquet("/Volumes/customer360analysis/customer/gold/feeds_data.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC At this point datasets for model training are ready and saved in goldlayer, the next step is gold layer transformation part 2, where we will use sql to create normalized tables that we will further use of data visualization and business insights.
# MAGIC The sql will be followed in next notebook.

# COMMAND ----------

