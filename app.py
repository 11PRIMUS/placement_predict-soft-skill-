from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

app = FastAPI(title="Placement Prediction System", description="Predict student placement based on soft skills scores")

# Mount static files only if directory exists
try:
    import os
    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    print(f"Static files not mounted: {e}")

templates = Jinja2Templates(directory="templates")

# Load prediction model with error handling
try:
    model = joblib.load('placement_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class PredictionRequest(BaseModel):
    mock_hr: float
    gd: float
    presentation: float
    english_cefr: int
    english_score: float

#response model
class PredictionResponse(BaseModel):
    prediction: str
    confidence: str
    result_class: str
    success: bool
    error: Optional[str] = None

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "message": "Placement Prediction API is running"
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request:Request,
    mock_hr:float= Form(...),
    gd:float = Form(...),
    presentation:float = Form(...),
    english_cefr:int = Form(...),
    english_score: float = Form(...)
):
    try:
        if model is None:
            raise Exception("Model not loaded")
        
        features = np.array([[mock_hr, gd, presentation, english_cefr, english_score]])
        
        #make prediction
        prediction =model.predict(features)[0]
        probability =model.predict_proba(features)[0]
        
        #confidenece percentage
        confidence =max(probability) * 100
        
        #result
        if prediction == 1:
            result="Likely to be PLACED"
            result_class = "success"
        else:
            result ="Unlikely to be placed"
            result_class= "danger"
            
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": result,
            "confidence": f"{confidence:.1f}%",
            "result_class": result_class,
            "show_result": True
        })
    
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error: {str(e)}"
        })

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_api(prediction_request: PredictionRequest):
    try:
        if model is None:
            raise Exception("Model not loaded")
            
        features = np.array([[
            prediction_request.mock_hr, 
            prediction_request.gd, 
            prediction_request.presentation, 
            prediction_request.english_cefr, 
            prediction_request.english_score
        ]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        confidence = max(probability) * 100
        
        if prediction == 1:
            result = "Likely to be PLACED"
            result_class = "success"
        else:
            result = "Unlikely to be placed"
            result_class = "danger"
            
        return PredictionResponse(
            prediction=result,
            confidence=f"{confidence:.1f}%",
            result_class=result_class,
            success=True
        )
    
    except Exception as e:
        return PredictionResponse(
            prediction="",
            confidence="",
            result_class="danger",
            success=False,
            error=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)