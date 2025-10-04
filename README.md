# Customer360Analysis

## Problem
Telecom companies face challenges in understanding their customers holistically. Customer data is siloed across contracts, services, demographics, behavior, and feedback. This fragmentation makes it difficult to predict churn, maximize customer lifetime value (CLTV), and improve satisfaction.

## Action
This project integrates multiple customer data sources to create a unified Customer 360 dataset. Advanced analytics and visualizations are developed to predict churn, segment customers, evaluate CLTV, and analyze satisfaction, supporting data-driven decision making.

## Tools
- **Databricks**: Performed EDA, Star Schema using SQL, feature engineering, and machine learning in databricks community edition.
- **Python & PySpark**: For ETL, data wrangling, feature engineering, and machine learning.
- **Power BI**: For business intelligence dashboards and interactive visualizations.
- **SQL**: For data extraction, transformation, and aggregation.
- **GitHub**: For version control and collaborative development.

## Approach
1. **Data Integration**: Consolidate data from contracts, services, demographics, behavior, and feedback tables.
2. **Feature Engineering**: Create derived features, such as tenure buckets, service counts, satisfaction scores, and financial indicators.
3. **Modeling & Analysis**:
   - Build churn prediction and CLTV regression models using PySpark.
   - Segment customers based on demographics, service usage, and satisfaction.
4. **Visualization & Reporting**: Develop Power BI dashboards for churn, CLTV, segmentation, and satisfaction analysis.
5. **Iterative Improvement**: Refine models and dashboards based on business feedback, and validate results using key performance metrics.

## Result and Outcome
- **Unified Customer 360 Dataset**: Enables comprehensive customer profiling and analytics.
- **Churn Prediction Model**: Identifies at-risk customers, allowing for targeted retention strategies.
- **CLTV Insights**: Reveals high-value customers and informs upsell/cross-sell campaigns.
- **Segmentation & Satisfaction Analysis**: Supports tailored marketing and service improvement.
- **Business Impact**: Improved retention, optimized marketing spend, and enhanced customer experience.

## Output/Result
- **Power BI Dashboard**: Interactive views for churn, CLTV, segmentation, satisfaction, and actionable business insights.
- **Model Performance**: 79% Churn prediction accuracy, CLTV regression metrics 82% R2 score, and segmentation results.
- **Data Artifacts**: Cleaned and integrated datasets, engineered features, and model outputs available in the repository.
- **Documentation**: Detailed data schema, process flows, and user guides included for ease of use and reproducibility.

---

_For details on schema, ETL scripts, modeling notebooks, and Power BI reports, see the respective folders and documentation in this repository._
