-- Databricks notebook source
-- MAGIC %md
-- MAGIC Creating star schema that will be helpful in data visualization

-- COMMAND ----------

-- Dimension: Demographics
CREATE OR REPLACE TABLE customer360analysis.gold.dim_customer_demographics
USING DELTA
AS
SELECT * FROM customer360analysis.silver.customer_demographics;

-- COMMAND ----------

-- Dimension: Services
CREATE OR REPLACE TABLE customer360analysis.gold.dim_customer_services
USING DELTA
AS
SELECT * FROM customer360analysis.silver.customer_services;

-- COMMAND ----------

-- Dimension: Contracts
CREATE OR REPLACE TABLE customer360analysis.gold.dim_customer_contracts
USING DELTA
AS
SELECT * FROM customer360analysis.silver.customer_contracts;


-- COMMAND ----------

-- Dimension: Feedback
CREATE OR REPLACE TABLE customer360analysis.gold.dim_customer_feedback
USING DELTA
AS
SELECT * FROM customer360analysis.silver.customer_feedback;

-- COMMAND ----------

-- Dimension: Behavior
CREATE OR REPLACE TABLE customer360analysis.gold.dim_customer_behavior
USING DELTA
AS
SELECT * FROM customer360analysis.silver.customer_behavior;

-- COMMAND ----------

-- Fact Table
CREATE OR REPLACE TABLE customer360analysis.gold.fact_customer_activity
USING DELTA
AS
SELECT
    c.customer_id,
    c.monthly_charge,
    c.total_charges,
    c.total_revenue,
    c.cltv,
    c.total_long_distance_charges,
    c.avg_monthly_long_distance_charges,
    c.total_extra_data_charges,
    c.total_refunds,
    f.churn,
    f.churn_score,
    f.satisfaction_score
FROM customer360analysis.silver.customer_contracts c
LEFT JOIN customer360analysis.silver.customer_feedback f
    ON c.customer_id = f.customer_id;

-- COMMAND ----------

show tables in customer360analysis.gold;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Performing quick sql exploratory analysis before jumping to visualization

-- COMMAND ----------

-- Total customers
SELECT COUNT(*) AS total_customers FROM gold_layer.customer_demographics;

-- Null check example
SELECT COUNT(*) AS null_zipcodes
FROM gold_layer.customer_demographics
WHERE zip_code IS NULL;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Demographics Overview

-- COMMAND ----------

-- Gender distribution
SELECT gender, COUNT(*) AS cnt
FROM gold_layer.customer_demographics
GROUP BY gender;


-- COMMAND ----------

-- Age buckets
SELECT 
  CASE 
    WHEN age < 30 THEN 'Under 30'
    WHEN age BETWEEN 30 AND 50 THEN '30-50'
    ELSE '50+' 
  END AS age_group,
  COUNT(*) AS cnt
FROM gold_layer.customer_demographics
GROUP BY age_group

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Revenue and Charges

-- COMMAND ----------

-- Average monthly charge per contract type
select c.contract, round(avg(c.monthly_charge),2) as avg_monthly_charge
from customer360analysis.gold.dim_customer_contracts c
group by c.contract;

-- COMMAND ----------

-- Top 5 states by total revenue
SELECT d.state, SUM(c.total_revenue) AS total_revenue
FROM customer360analysis.gold.dim_customer_contracts c
JOIN customer360analysis.gold.dim_customer_demographics d
  ON c.customer_id = d.customer_id
GROUP BY d.state
ORDER BY total_revenue DESC
LIMIT 5;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Churn Analysis

-- COMMAND ----------

-- Overall churn rate
SELECT 
  ROUND(100 * SUM(CASE WHEN churn = 1 THEN 1 ELSE 0 END) / COUNT(*),2) AS churn_rate_pct
FROM customer360analysis.gold.dim_customer_feedback;

-- COMMAND ----------

-- Churn by contract type
SELECT c.contract, 
       ROUND(100 * SUM(CASE WHEN f.churn = 1 THEN 1 ELSE 0 END) / COUNT(*),2) AS churn_rate_pct
FROM customer360analysis.gold.dim_customer_contracts c
JOIN customer360analysis.gold.dim_customer_feedback f
  ON c.customer_id = f.customer_id
GROUP BY c.contract;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC satisfaction and referrals

-- COMMAND ----------

-- Average satisfaction score by internet type
SELECT s.internet_type, ROUND(AVG(f.satisfaction_score),2) AS avg_satisfaction
FROM customer360analysis.gold.dim_customer_services s
JOIN customer360analysis.gold.dim_customer_feedback f
  ON s.customer_id = f.customer_id
GROUP BY s.internet_type
ORDER BY avg_satisfaction DESC;

-- COMMAND ----------

-- Referrals vs churn
SELECT f.number_of_referrals, 
       ROUND(100 * SUM(CASE WHEN f.churn = 1 THEN 1 ELSE 0 END) / COUNT(*),2) AS churn_rate_pct
FROM customer360analysis.gold.dim_customer_feedback f
GROUP BY f.number_of_referrals
ORDER BY f.number_of_referrals;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Behavior Insights

-- COMMAND ----------

-- Avg data usage by churn status
SELECT f.churn, ROUND(AVG(b.avg_monthly_gb_download),2) AS avg_gb_download
FROM customer360analysis.gold.dim_customer_behavior b
JOIN customer360analysis.gold.dim_customer_feedback f
  ON b.customer_id = f.customer_id
GROUP BY f.churn;

-- COMMAND ----------

-- Tenure vs churn
SELECT 
  CASE 
    WHEN b.tenure_in_months < 12 THEN '<1 year'
    WHEN b.tenure_in_months BETWEEN 12 AND 24 THEN '1-2 years'
    ELSE '2+ years'
  END AS tenure_group,
  ROUND(100 * SUM(CASE WHEN f.churn = 1 THEN 1 ELSE 0 END) / COUNT(*),2) AS churn_rate_pct
FROM customer360analysis.gold.dim_customer_behavior b
JOIN customer360analysis.gold.dim_customer_feedback f
  ON b.customer_id = f.customer_id
GROUP BY tenure_group
ORDER BY tenure_group;

-- COMMAND ----------

select city, country from customer360analysis.gold.dim_customer_demographics;

-- COMMAND ----------

select satisfaction_score from customer360analysis.gold.dim_customer_feedback


-- COMMAND ----------

