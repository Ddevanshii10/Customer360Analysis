<<<<<<< HEAD
# Customer 360 Analytics Platform

A comprehensive customer analytics platform built with Flask, featuring advanced churn prediction, CLTV analysis, customer segmentation, and feedback analysis for the telecom industry.

## 🚀 Features

- **📊 Customer Analytics Overview** - Real-time dashboard with key metrics
- **🎯 Churn Prediction** - ML-powered customer churn prediction with 95% accuracy
- **💰 CLTV Analysis** - Customer Lifetime Value calculation and forecasting
- **🧩 Customer Segmentation** - Intelligent clustering for targeted campaigns
- **💬 Feedback Analysis** - Sentiment analysis and satisfaction trends

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **ML Models**: Scikit-learn, Pandas, NumPy
- **Visualization**: Plotly
- **Database**: CSV files (can be extended to SQL databases)

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/customer360-analytics.git
   cd customer360-analytics
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**
   
   On Windows:
   ```bash
   .\.venv\Scripts\Activate.ps1
   ```
   
   On macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

1. **Start the Flask application**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to**
   ```
   http://127.0.0.1:5000
   ```

3. **Access the Dashboard**
   ```
   http://127.0.0.1:5000/dashboard
   ```

## 📈 Usage

### Churn Prediction
- Enter customer tenure, monthly charges, total charges, and contract type
- Get instant churn probability and risk level assessment
- Receive actionable recommendations for customer retention

### CLTV Analysis
- Input customer data and service type
- Calculate predicted Customer Lifetime Value
- Get value category classification and strategic recommendations

### Customer Segmentation
- Analyze customer demographics and behavior
- Identify customer segments automatically
- Get targeted marketing strategy suggestions

### Feedback Analysis
- Enter customer feedback text and ratings
- Get sentiment analysis results
- Receive action recommendations based on sentiment

## 📁 Project Structure

```
Customer360Analysis/
├── app.py                 # Main Flask application
├── templates/            # HTML templates
│   ├── index.html        # Landing page
│   ├── dashboard.html    # Main dashboard
│   └── static/          # CSS, JS, and images
├── requirements.txt      # Python dependencies
├── *.pkl                # ML model files
├── *.csv                # Sample data files
└── README.md            # This file
```

## 🎯 API Endpoints

- `GET /` - Landing page
- `GET /dashboard` - Main dashboard
- `POST /predict_churn` - Churn prediction API
- `POST /predict_cltv` - CLTV prediction API
- `POST /segment_customer` - Customer segmentation API

## 🔧 Configuration

The application uses Flask's development server by default. For production deployment:

1. Set environment variables
2. Configure a production WSGI server (e.g., Gunicorn)
3. Set up a proper database
4. Configure SSL/HTTPS

## 📊 Sample Data

The application includes sample CSV files for demonstration:
- `Chrun.csv` - Customer churn data
- `CLTV.csv` - Customer lifetime value data
- `segment.csv` - Customer segmentation data
- `feedback.csv` - Customer feedback data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Devanshi**
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- Flask community for excellent documentation
- Scikit-learn for machine learning capabilities
- Plotly for beautiful visualizations

---

⭐ If you found this project helpful, please give it a star!
=======
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
>>>>>>> 5805f994cac284de412f2899b260bb5a0704dd2f
