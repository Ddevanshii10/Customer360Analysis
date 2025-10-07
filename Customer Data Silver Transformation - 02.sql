-- Databricks notebook source
-- Create schema if it doesn’t exist
CREATE SCHEMA IF NOT EXISTS customer360analysis.silver;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Bringing telco and feedback table from bronze schema to silver to create normalized tables 

-- COMMAND ----------

CREATE OR REPLACE TABLE customer360analysis.silver.customer_demographics
USING DELTA
AS
SELECT
  customer_id,
  age,
  gender,
  married,
  senior_citizen,
  country,
  state,
  city,
  zip_code,
  under_30,
  lat_long,
  latitude,
  longitude,
  population
FROM customer360analysis.bronze.telco;

-- COMMAND ----------

CREATE oR REPLACE TABLE customer360analysis.silver.customer_services
USING DELTA
AS
SELECT
  customer_id,
  phone_service,
  multiple_lines,
  internet_service,
  internet_type,
  streaming_tv,
  streaming_movies,
  streaming_music,
  device_protection_plan,
  premium_tech_support,
  online_security,
  online_backup,
  unlimited_data,
  contract
FROM customer360analysis.bronze.telco;

-- COMMAND ----------

CREATE OR REPLACE TABLE customer360analysis.silver.customer_contracts
USING DELTA
AS
SELECT
  customer_id,
  contract,
  tenure_in_months,
  offer,
  payment_method,
  paperless_billing,
  monthly_charge,
  total_charges,
  total_revenue,
  cltv,
  total_long_distance_charges,
  avg_monthly_long_distance_charges,
  total_extra_data_charges,
  total_refunds
FROM customer360analysis.bronze.telco;

-- COMMAND ----------

CREATE OR REPLACE TABLE customer360analysis.silver.customer_feedback
USING DELTA
AS
SELECT
  customer_id,
  satisfaction_score,
  number_of_referrals,
  referred_a_friend,
  churn,
  churn_Score,
  churn_category,
  churn_reason,
  partner,
  churnreason_missing
FROM customer360analysis.bronze.telco;

-- COMMAND ----------

-- Customer Behavior
CREATE OR REPLACE TABLE customer360analysis.silver.customer_behavior
USING DELTA
AS
SELECT
  customer_id,
  tenure_in_months,
  quarter,
  offer,
  number_of_referrals,
  referred_a_friend,
  avg_monthly_gb_download
FROM customer360analysis.bronze.telco;

-- COMMAND ----------

show tables in customer360analysis.silver;

-- COMMAND ----------

