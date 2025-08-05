# fastapi_app/main.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ComplaintInput(BaseModel):
    agency_name: str
    borough: str
    survey_month: str
    dissatisfaction_reason: str
    justified_reason: str

@app.post("/predict")
def predict_sentiment(data: ComplaintInput):
    text = data.dissatisfaction_reason.lower()

    if any(word in text for word in ["bad", "ignored", "rude", "slow", "not working"]):
        sentiment = "negative"
    elif any(word in text for word in ["good", "helpful", "quick", "polite", "great"]):
        sentiment = "positive"
    else:
        sentiment = "neutral"

    return {"prediction": sentiment}


