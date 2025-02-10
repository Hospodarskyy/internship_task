from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib

# Load saved Naïve Bayes model & vectorizer
model = joblib.load("nb_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()

class TextRequest(BaseModel):
    texts: List[str]

@app.post("/analyze")
def analyze_texts(request: TextRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty")
    
    # Identify empty texts and notify the user
    non_empty_texts = [text for text in request.texts if text.strip()]
    empty_texts = [text for text in request.texts if not text.strip()]

    notification = None
    if empty_texts:
        notification = f"{len(empty_texts)} empty text(s) were skipped."
    
    # Convert non-empty texts to features
    X_input = vectorizer.transform(non_empty_texts)

    # Predict sentiment
    predictions = model.predict(X_input)

    # Ensure predictions are serializable
    results = [{"text": text, "sentiment": str(pred)} for text, pred in zip(non_empty_texts, predictions)]
    
    # Include notification if any texts were empty
    response = {"results": results}
    if notification:
        response["notification"] = notification
    
    return response
