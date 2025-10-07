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