from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Annotated, Literal, Optional
import pickle
import numpy as np
import pandas as pd

# Load the ML model and vectorizer
try:
    with open('password_strength_XGBoost_0.9405.pkl', 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    vectorizer = model_package['vectorizer']
    print(f"✅ Model loaded: {model_package['model_name']} with accuracy {model_package['accuracy']:.4f}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    vectorizer = None

app = FastAPI()

class PasswordInput(BaseModel):
    password: Annotated[str, Field(..., min_length=1, description='The password to analyze')]
    
    @computed_field
    @property
    def length(self) -> int:
        return len(self.password)
    
    @computed_field
    @property
    def lowercase_freq(self) -> float:
        if len(self.password) == 0:
            return 0
        return len([c for c in self.password if c.islower()]) / len(self.password)
    
    @computed_field
    @property
    def has_uppercase(self) -> bool:
        return any(c.isupper() for c in self.password)
    
    @computed_field
    @property
    def has_digit(self) -> bool:
        return any(c.isdigit() for c in self.password)
    
    @computed_field
    @property
    def has_special(self) -> bool:
        return any(not c.isalnum() for c in self.password)

class PasswordResponse(BaseModel):
    password: str
    strength: Literal['weak', 'medium', 'strong']
    confidence: float
    class_probabilities: dict
    features: dict

@app.get("/")
def root():
    return {
        'service': '🔐 Password Strength Classifier API',
        'status': 'active',
        'model_loaded': model is not None
    }

@app.get("/health")
def health_check():
    if model is None:
        return JSONResponse(
            status_code=503,
            content={'status': 'unhealthy', 'message': 'Model not loaded'}
        )
    return {'status': 'healthy', 'model': 'loaded'}

@app.post("/predict", response_model=PasswordResponse)
def predict_password_strength(data: PasswordInput):
    """
    Predict the strength of a password
    - Returns: weak, medium, or strong classification
    - Includes confidence scores and feature analysis
    """
    if model is None:
        return JSONResponse(
            status_code=503,
            content={'error': 'Model not loaded. Please check server logs.'}
        )
    
    try:
        # Get TF-IDF features
        password_array = np.array([data.password])
        tfidf_matrix = vectorizer.transform(password_array)
        
        # Get structural features
        length = data.length
        lowercase_freq = data.lowercase_freq
        
        # Combine features
        tfidf_array = tfidf_matrix.toarray()
        combined = np.append(tfidf_array, [[length, lowercase_freq]], axis=1)
        
        # Predict
        pred = model.predict(combined)[0]
        proba = model.predict_proba(combined)[0]
        
        # Map prediction to label
        strength_map = {0: 'weak', 1: 'medium', 2: 'strong'}
        predicted_strength = strength_map[pred]
        
        # Create class probabilities dictionary
        prob_dict = {
            'weak': float(proba[0]),
            'medium': float(proba[1]),
            'strong': float(proba[2])
        }
        
        # Feature summary
        features = {
            'length': length,
            'lowercase_freq': lowercase_freq,
            'has_uppercase': data.has_uppercase,
            'has_digit': data.has_digit,
            'has_special': data.has_special
        }
        
        response = PasswordResponse(
            password=data.password,
            strength=predicted_strength,
            confidence=float(max(proba)),
            class_probabilities=prob_dict,
            features=features
        )
        
        return response
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'error': f'Prediction failed: {str(e)}'}
        )

