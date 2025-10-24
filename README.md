# Placement Prediction Web Application

A modern FastAPI web application that predicts student placement probability based on their soft skills scores.

## 🚀 Live Demo
**Deployed on Render:** [https://placement-softv2.onrender.com]

## Features
- User-friendly web interface with Bootstrap styling
- Real-time placement prediction
- Confidence score display
- REST API endpoints for programmatic access
- Interactive API demo
- Responsive design
- Automatic API documentation

## Required Input Scores:
1. **Mock HR Interview Score** (5-20)
2. **Group Discussion Score** (4-20) 
3. **Presentation Score** (5-20)
4. **English CEFR Level** (0=A1, 1=A2, 2=B1, 3=B2/C1)
5. **English Score** (5-20)

## Local Development:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Access at: http://localhost:8000
``
## Project Structure:
```
├── app.py                    # Main FastAPI application
├── templates/index.html      # Web interface
├── static/                   # Static files (if any)
├── placement_model.pkl       # Trained ML model
├── requirements.txt          # Python dependencies
└── Procfile                  # Render deployment config
```