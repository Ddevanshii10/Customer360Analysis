// Dashboard Navigation
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard JS loaded');
    
    // Check if buttons exist
    const buttons = document.querySelectorAll('.sidebar-btn');
    console.log('Found buttons:', buttons.length);
    
    buttons.forEach(btn => {
        btn.addEventListener('click', function() {
            console.log('Button clicked:', this.getAttribute('data-section'));
            
            // Remove active class from all buttons and sections
            document.querySelectorAll('.sidebar-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.dashboard-section').forEach(s => s.classList.remove('active'));
            
            // Add active class to clicked button
            this.classList.add('active');
            
            // Show corresponding section
            const sectionId = this.getAttribute('data-section');
            const targetSection = document.getElementById(sectionId);
            console.log('Target section:', targetSection);
            
            if (targetSection) {
                targetSection.classList.add('active');
                console.log('Section activated:', sectionId);
            } else {
                console.error('Section not found:', sectionId);
            }
        });
    });
    
    // Load overview charts
    loadOverviewCharts();
});

function loadOverviewCharts() {
    fetch('/generate_insights')
        .then(response => response.json())
        .then(data => {
            if (data.churn_chart) {
                Plotly.newPlot('churn-overview-chart', JSON.parse(data.churn_chart).data, JSON.parse(data.churn_chart).layout);
            }
            if (data.cltv_chart) {
                Plotly.newPlot('cltv-overview-chart', JSON.parse(data.cltv_chart).data, JSON.parse(data.cltv_chart).layout);
            }
        })
        .catch(error => console.error('Error loading charts:', error));
}

// Churn Prediction Form
document.getElementById('churn-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const data = Object.fromEntries(formData);
    
    fetch('/predict_churn', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        if (result.error) {
            alert('Error: ' + result.error);
            return;
        }
        
        const resultDiv = document.getElementById('churn-result');
        const contentDiv = document.getElementById('churn-prediction-content');
        
        const riskColor = result.risk_level === 'High' ? '#e74c3c' : 
                         result.risk_level === 'Medium' ? '#f39c12' : '#27ae60';
        
        contentDiv.innerHTML = `
            <div class="prediction-result">
                <div class="result-item">
                    <strong>Churn Prediction:</strong> ${result.churn_prediction === 1 ? 'Will Churn' : 'Will Not Churn'}
                </div>
                <div class="result-item">
                    <strong>Churn Probability:</strong> ${(result.churn_probability * 100).toFixed(2)}%
                </div>
                <div class="result-item">
                    <strong>Risk Level:</strong> <span style="color: ${riskColor}; font-weight: bold;">${result.risk_level}</span>
                </div>
            </div>
        `;
        
        resultDiv.style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while predicting churn.');
    });
});

// CLTV Prediction Form
document.getElementById('cltv-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const data = Object.fromEntries(formData);
    
    fetch('/predict_cltv', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        if (result.error) {
            alert('Error: ' + result.error);
            return;
        }
        
        const resultDiv = document.getElementById('cltv-result');
        const contentDiv = document.getElementById('cltv-prediction-content');
        
        const valueColor = result.value_category === 'High Value' ? '#27ae60' : 
                          result.value_category === 'Medium Value' ? '#f39c12' : '#e74c3c';
        
        contentDiv.innerHTML = `
            <div class="prediction-result">
                <div class="result-item">
                    <strong>Predicted CLTV:</strong> $${result.cltv_prediction.toFixed(2)}
                </div>
                <div class="result-item">
                    <strong>Value Category:</strong> <span style="color: ${valueColor}; font-weight: bold;">${result.value_category}</span>
                </div>
            </div>
        `;
        
        resultDiv.style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while predicting CLTV.');
    });
});

// Customer Segmentation Form
document.getElementById('segment-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const data = Object.fromEntries(formData);
    
    fetch('/segment_customer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        if (result.error) {
            alert('Error: ' + result.error);
            return;
        }
        
        const resultDiv = document.getElementById('segment-result');
        const contentDiv = document.getElementById('segment-prediction-content');
        
        const recommendationsList = result.recommendations.map(rec => `<li>${rec}</li>`).join('');
        
        contentDiv.innerHTML = `
            <div class="prediction-result">
                <div class="result-item">
                    <strong>Customer Segment:</strong> ${result.segment_name}
                </div>
                <div class="result-item">
                    <strong>Segment ID:</strong> ${result.segment_id}
                </div>
                <div class="result-item">
                    <strong>Recommendations:</strong>
                    <ul style="margin-top: 10px; padding-left: 20px;">
                        ${recommendationsList}
                    </ul>
                </div>
            </div>
        `;
        
        resultDiv.style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while segmenting customer.');
    });
});

// Feedback Analysis Form
document.getElementById('feedback-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const data = Object.fromEntries(formData);
    
    fetch('/analyze_feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        if (result.error) {
            alert('Error: ' + result.error);
            return;
        }
        
        const resultDiv = document.getElementById('feedback-result');
        const contentDiv = document.getElementById('feedback-prediction-content');
        
        const satisfactionColor = result.satisfaction_level.includes('Satisfied') ? '#27ae60' : 
                                 result.satisfaction_level === 'Neutral' ? '#f39c12' : '#e74c3c';
        
        contentDiv.innerHTML = `
            <div class="prediction-result">
                <div class="result-item">
                    <strong>Satisfaction Level:</strong> <span style="color: ${satisfactionColor}; font-weight: bold;">${result.satisfaction_level}</span>
                </div>
                <div class="result-item">
                    <strong>Feedback Cluster:</strong> ${result.feedback_cluster}
                </div>
                <div class="result-item">
                    <strong>Action Required:</strong> ${result.action_required ? 'Yes - Immediate attention needed' : 'No - Customer is satisfied'}
                </div>
            </div>
        `;
        
        resultDiv.style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while analyzing feedback.');
    });
});

// Add CSS for prediction results
const style = document.createElement('style');
style.textContent = `
    .prediction-result {
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    .result-item {
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    .result-item:last-child {
        margin-bottom: 0;
    }
    
    .result-item strong {
        color: #2c3e50;
    }
`;
document.head.appendChild(style);