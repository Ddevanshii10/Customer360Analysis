-- Databricks notebook source
-- MAGIC %md
-- MAGIC Making bronze, silver and gold schemas in customer360analysis

-- COMMAND ----------

CREATE SCHEMA IF NOT EXISTS customer360analysis.bronze;
CREATE SCHEMA IF NOT EXISTS customer360analysis.silver;
CREATE SCHEMA IF NOT EXISTS customer360analysis.gold;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##### converting my telco and feeds data from parquet to sql so that we can create tables

-- COMMAND ----------

-- MAGIC %python
-- MAGIC #Example for Telco
-- MAGIC df_telco = spark.read.parquet("/Volumes/customer360analysis/customer/silver/huggingface_telco_silver.parquet/")
-- MAGIC
-- MAGIC # Example for Feedback
-- MAGIC df_feedback = spark.read.parquet("/Volumes/customer360analysis/customer/silver/feedback_satisfaction_silver.parquet/")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Function to clean column names
-- MAGIC def clean_column_names(df):
-- MAGIC     for col_name in df.columns:
-- MAGIC         new_col = col_name.strip().lower().replace(" ", "_").replace("-", "_")
-- MAGIC         new_col = (
-- MAGIC             new_col.replace(",", "")
-- MAGIC             .replace(";", "")
-- MAGIC             .replace("{", "")
-- MAGIC             .replace("}", "")
-- MAGIC             .replace("(", "")
-- MAGIC             .replace(")", "")
-- MAGIC             .replace("\n", "")
-- MAGIC             .replace("\t", "")
-- MAGIC             .replace("=", "")
-- MAGIC         )
-- MAGIC         df = df.withColumnRenamed(col_name, new_col)
-- MAGIC     return df
-- MAGIC
-- MAGIC # Clean column names
-- MAGIC df_telco = clean_column_names(df_telco)
-- MAGIC df_feedback = clean_column_names(df_feedback)
-- MAGIC
-- MAGIC # Create the schema if it does not exist
-- MAGIC spark.sql("CREATE SCHEMA IF NOT EXISTS silver_layer")
-- MAGIC
-- MAGIC # Save as Delta in Silver Layer
-- MAGIC df_telco.write.format("delta").mode("overwrite").saveAsTable("customer360analysis.bronze.telco")
-- MAGIC df_feedback.write.format("delta").mode("overwrite").saveAsTable("customer360analysis.bronze.feedback")

-- COMMAND ----------

SHOW TABLES IN customer360analysis.bronze;


-- COMMAND ----------

-- MAGIC %python
-- MAGIC df_telco.columns

-- COMMAND ----------



-- COMMAND ----------




-- COMMAND ----------




-- COMMAND ----------




-- COMMAND ----------



-- COMMAND ----------



-- COMMAND ----------



-- COMMAND ----------




-- COMMAND ----------











-- COMMAND ----------



-- COMMAND ----------



-- COMMAND ----------



-- COMMAND ----------



-- COMMAND ----------



-- COMMAND ----------



-- COMMAND ----------



-- COMMAND ----------

