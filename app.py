from flask import Flask, render_template, request, jsonify
import pickle
import joblib
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__, template_folder='templates', static_folder='templates/static')

# Load your pre-trained models
churn_model = None
cltv_model = None
segment_model = None
feedback_model = None

try:
    churn_model = pickle.load(open('churn_model.pkl', 'rb'))
    print(f"Churn model loaded: {type(churn_model)}")
    print(f"Churn model has predict method: {hasattr(churn_model, 'predict')}")
except Exception as e:
    print(f"Error loading churn model: {e}")

try:
    cltv_model = pickle.load(open('cltv_model.pkl', 'rb'))
    print(f"CLTV model loaded: {type(cltv_model)}")
    print(f"CLTV model has predict method: {hasattr(cltv_model, 'predict')}")
except Exception as e:
    print(f"Error loading CLTV model: {e}")

try:
    segment_model = joblib.load('segment_kmeans_model.joblib')
    print(f"Segment model loaded: {type(segment_model)}")
except Exception as e:
    print(f"Error loading segment model: {e}")

try:
    feedback_model = joblib.load('feedback_kmeans_model.joblib')
    print(f"Feedback model loaded: {type(feedback_model)}")
except Exception as e:
    print(f"Error loading feedback model: {e}")

print("Model loading completed!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    try:
        if churn_model is None:
            return jsonify({'error': 'Churn model not loaded'}), 500
            
        data = request.json
        print("Received churn data:", data)
        
        # Create features array with proper shape
        features = np.array([[
            float(data.get('tenure', 0)),
            float(data.get('monthly_charges', 0)),
            float(data.get('total_charges', 0))
        ]])
        
        print("Churn features shape:", features.shape)
        print("Churn model type:", type(churn_model))
        
        # Handle different model types
        if hasattr(churn_model, 'predict'):
            # It's a proper sklearn model
            prediction = churn_model.predict(features)[0]
            if hasattr(churn_model, 'predict_proba'):
                probability = churn_model.predict_proba(features)[0][1]
            else:
                probability = float(prediction)
        else:
            # Fallback for numpy arrays or other formats
            print("Model doesn't have predict method, using fallback")
            # Simple rule-based prediction as fallback
            monthly_charges = float(data.get('monthly_charges', 0))
            tenure = float(data.get('tenure', 0))
            contract_type = data.get('contract_type', '')
            
            # Simple heuristic
            risk_score = 0
            if monthly_charges > 80: risk_score += 0.3
            if tenure < 12: risk_score += 0.4
            if contract_type == 'Month-to-month': risk_score += 0.3
            
            prediction = 1 if risk_score > 0.5 else 0
            probability = min(risk_score, 0.9)
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'
        })
        
    except Exception as e:
        print("Churn prediction error:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/predict_cltv', methods=['POST'])
def predict_cltv():
    try:
        if cltv_model is None:
            return jsonify({'error': 'CLTV model not loaded'}), 500
            
        data = request.json
        print("Received CLTV data:", data)
        
        # Create features array
        features = np.array([[
            float(data.get('tenure', 0)),
            float(data.get('monthly_charges', 0)),
            float(data.get('total_charges', 0))
        ]])
        
        print("CLTV features shape:", features.shape)
        print("CLTV model type:", type(cltv_model))
        
        # Handle different model types
        if hasattr(cltv_model, 'predict'):
            # It's a proper sklearn model
            cltv_prediction = cltv_model.predict(features)[0]
        else:
            # Fallback calculation
            print("CLTV model doesn't have predict method, using fallback")
            monthly_charges = float(data.get('monthly_charges', 0))
            tenure = float(data.get('tenure', 0))
            service_type = data.get('service_type', 'Basic')
            
            # Simple CLTV calculation: Monthly charges * expected lifetime * service multiplier
            service_multiplier = {'Premium': 1.5, 'Standard': 1.2, 'Basic': 1.0}.get(service_type, 1.0)
            expected_lifetime = max(tenure * 2, 24)  # At least 2 years
            cltv_prediction = monthly_charges * expected_lifetime * service_multiplier
        
        return jsonify({
            'cltv': float(cltv_prediction),
            'value_category': 'High Value' if cltv_prediction > 5000 else 'Medium Value' if cltv_prediction > 2000 else 'Low Value'
        })
        
    except Exception as e:
        print("CLTV prediction error:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/segment_customer', methods=['POST'])
def segment_customer():
    try:
        data = request.json
        # Assuming your segmentation model expects customer behavior features
        features = np.array([[
            float(data.get('recency', 0)),
            float(data.get('frequency', 0)),
            float(data.get('monetary', 0)),
            float(data.get('tenure', 0))
        ]])
        
        segment = segment_model.predict(features)[0]
        
        segment_names = {
            0: 'Champions',
            1: 'Loyal Customers',
            2: 'Potential Loyalists',
            3: 'At Risk',
            4: 'Cannot Lose Them'
        }
        
        return jsonify({
            'segment_id': int(segment),
            'segment_name': segment_names.get(segment, 'Unknown'),
            'recommendations': get_segment_recommendations(segment)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/analyze_feedback', methods=['POST'])
def analyze_feedback():
    try:
        data = request.json
        # Assuming your feedback model expects sentiment and rating features
        features = np.array([[
            float(data.get('sentiment_score', 0)),
            float(data.get('rating', 0)),
            float(data.get('response_time', 0)),
            float(data.get('resolution_time', 0))
        ]])
        
        feedback_cluster = feedback_model.predict(features)[0]
        
        cluster_names = {
            0: 'Highly Satisfied',
            1: 'Satisfied',
            2: 'Neutral',
            3: 'Dissatisfied',
            4: 'Highly Dissatisfied'
        }
        
        return jsonify({
            'feedback_cluster': int(feedback_cluster),
            'satisfaction_level': cluster_names.get(feedback_cluster, 'Unknown'),
            'action_required': feedback_cluster >= 3
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/generate_insights')
def generate_insights():
    # Generate sample visualizations for the dashboard
    try:
        # Sample data for demonstration
        sample_data = generate_sample_data()
        
        # Create charts
        churn_chart = create_churn_chart(sample_data)
        cltv_chart = create_cltv_chart(sample_data)
        segment_chart = create_segment_chart(sample_data)
        
        return jsonify({
            'churn_chart': churn_chart,
            'cltv_chart': cltv_chart,
            'segment_chart': segment_chart
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_segment_recommendations(segment):
    recommendations = {
        0: ["Reward them for their loyalty", "Ask for referrals", "Upsell premium products"],
        1: ["Keep them engaged", "Offer loyalty programs", "Regular communication"],
        2: ["Offer membership programs", "Recommend products", "Special offers"],
        3: ["Re-engagement campaigns", "Personalized offers", "Win-back strategies"],
        4: ["Immediate attention required", "Personal contact", "Exclusive offers"]
    }
    return recommendations.get(segment, ["General customer care"])

def generate_sample_data():
    np.random.seed(42)
    n = 1000
    return {
        'customer_id': range(1, n+1),
        'churn': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'cltv': np.random.exponential(500, n),
        'segment': np.random.choice([0, 1, 2, 3, 4], n),
        'tenure': np.random.randint(1, 60, n),
        'monthly_charges': np.random.normal(70, 20, n)
    }

def create_churn_chart(data):
    df = pd.DataFrame(data)
    churn_counts = df['churn'].value_counts()
    
    fig = px.pie(values=churn_counts.values, 
                 names=['Retained', 'Churned'],
                 title='Customer Churn Distribution')
    
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def create_cltv_chart(data):
    df = pd.DataFrame(data)
    
    fig = px.histogram(df, x='cltv', nbins=30, 
                      title='Customer Lifetime Value Distribution')
    
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def create_segment_chart(data):
    df = pd.DataFrame(data)
    segment_counts = df['segment'].value_counts()
    segment_names = ['Champions', 'Loyal', 'Potential', 'At Risk', 'Cannot Lose']
    
    fig = px.bar(x=segment_names, y=segment_counts.values,
                title='Customer Segmentation')
    
    return json.dumps(fig, cls=PlotlyJSONEncoder)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)