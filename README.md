# Customer360Analysis


Customer Segmentation, CLTV Prediction, and Churn Analysis using Machine Learning & Power BI

##  Problem Statement

In today’s competitive telecom and internet service industry, retaining customers and maximizing their lifetime value (CLTV) is crucial. The company faced challenges in:

Identifying high-value customers.

Understanding customer behavior patterns.

Predicting churn risk.

Segmenting customers for action-oriented strategies (retention, upselling) and behavioral analysis (demographics, service usage).

Without clear insights, marketing and retention strategies were reactive rather than proactive.

##  Objectives

Predict Customer Lifetime Value (CLTV) with high accuracy.

Segment customers into meaningful groups based on action-oriented and behavioral perspectives.

Analyze churn drivers to design effective retention campaigns.

Build an interactive Customer 360° dashboard in Power BI for decision-makers.

##  Data Collection & Preparation

Data Sources: Customer demographics, contracts, services, usage behavior, satisfaction scores, and activity history.

Data Cleaning: Handled missing values, duplicates, and outliers.

Feature Engineering: Created CLTV ranges, satisfaction groups, churn labels, and referral insights.

Schema Design: Structured into fact and dimension tables (fact_customer_activity, dim_customer_behavior, dim_customer_contracts, dim_customer_demographics, etc.).

## Methodology
1. Model Development

Built a CLTV prediction model using regression algorithms.

Achieved R² score of 82%, ensuring robust predictive accuracy.

2. Customer Segmentation

Action-Oriented Segmentation: Combined CLTV and satisfaction score to create categories such as Delight & Grow, Retention Needed, Nurture, Low Value.

Behavioral Segmentation: Grouped customers by demographics (age, gender, city), usage behavior (streaming, downloads, referrals), and service adoption.

3. Churn Analysis

Applied AI-powered visuals in Power BI (Key Influencers, Decomposition Tree) to identify churn drivers.

Found that tenure, internet type, contract duration, and satisfaction score were the most important churn factors.

4. Visualization (Power BI Customer 360 Dashboard)

Created multiple interactive pages:

Action-Oriented Customer Segmentation (focus on profitability and retention strategies).

Behavioral Segmentation (focus on demographics, city distribution, gender, and service adoption).

Churn Analysis (key influencers, heatmaps, decomposition tree).

CLTV Analysis (customer value distribution, revenue insights).

## Key Findings & Business Insights

High CLTV Customers with High Satisfaction → Best candidates for upselling premium services (Delight & Grow).

High CLTV but Low Satisfaction → Require retention efforts like loyalty programs or proactive support (Retention Needed).

Low CLTV with High Satisfaction → Can be nurtured to grow their value (Nurture).

Low CLTV & Low Satisfaction → Low-value customers, less priority for investments.

Churn Drivers: DSL customers, month-to-month contracts, and customers with tenure <12 months showed higher churn risk.

Behavioral Insights:

Majority of customers aged 30–50.

Fiber optic users had higher CLTV but also higher refund requests.

Gender distribution was almost equal, but women showed slightly higher satisfaction.

## Business Impact

Enabled data-driven segmentation for marketing campaigns.

Identified retention-needed customers to reduce churn.

Unlocked opportunities for cross-sell and upsell with Delight & Grow segment.

Delivered a single Customer 360° dashboard integrating predictive analytics and business intelligence.

## Conclusion

The project successfully combined machine learning (CLTV prediction) with business intelligence (Power BI dashboards) to provide a holistic view of customer value and churn. The integration of AI-powered visuals further enhanced the ability of managers to explore and act on insights.

## Future Scope

Deploy real-time dashboards integrated with CRM systems.

Extend churn prediction into a classification ML model with live scoring.

Incorporate social media sentiment analysis for better satisfaction tracking.

Automate personalized campaign recommendations for each customer segment.
