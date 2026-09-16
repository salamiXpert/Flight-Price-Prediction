"""
serve_api_class.py - practical, lightweight API server for the trained flight price model.

This script does ONE thing: 
load the already-saved model - "flight_price_model.pkl" and serve prediction over HTTP

PREREQUISITE:
'flight_price_model.pkl' - this should be in the project folder

RUN: python serve_api_class.py
(If successful)
THEN: Open http://0.0.0.0:8000/docs on your browser and change it to  http://localhost:8000/docs

Assignment:
 YOUR EXCHANGE API = https://v6.exchangerate-api.com/v6/YOUR-API-KEY/pair/INR/NGN
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

import joblib
import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_PATH = "flight_price_model.pkl"
FEEDBACK_FILE = "incoming_feedback_data.csv"
API_KEY = os.getenv("API_KEY")

if not os.path.exists(MODEL_PATH):
    sys.exit(
        f"'{MODEL_PATH}' not found in this folder.\n"
        "Run flight_price_prediction.py (or the notebook) first to train"
        "and save the model, then re-run this script"
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(title = "Flight Price Prediction API")

def get_inr_to_ngn_rate() -> float:
    """Fetch a live INR->NGN rate; fall back to a fixed rate if the request fails."""
    try:
        response = requests.get("https://v6.exchangerate-api.com/v6/2d9809d1a28c101a49d1a7f1/pair/INR/NGN", timeout=3)
        response.raise_for_status()
        return float(response.json()["rates"]["NGN"])
    except Exception as exc:
        print(f"Could not fetch live rate ({exc}); using fallback rate.")
        return 18.50 # fallback conversion rate


class FlightInput(BaseModel):
    airline: str
    from_city: str
    to_city: str
    travel_class: str
    stop: str
    days_of_week: str
    dep_time_block: str
    arr_time_block: str
    Duration_mins: float
    days_left: int
    actual_price: Optional[float] = None

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Flight Price Prediction API is running"}

@app.post("/predict")
def predict(data: FlightInput):
    input_dict = {
        "airline": [data.airline],
        "from": [data.from_city],
        "to": [data.to_city],
        "class": [data.travel_class],
        "stop": [data.stop],
        "days_of_week": [data.days_of_week],
        "dep_time_block": [data.dep_time_block],
        "arr_time_block": [data.arr_time_block],
        "Duration_mins": [data.Duration_mins],
        "days_left": [data.days_left],
    }
    df_input = pd.DataFrame(input_dict)

    try:
        predicted_inr = float(model.predict(df_input)[0])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not score this input: {exc}")

    ngn_rate = get_inr_to_ngn_rate()
    predicted_ngn = round(predicted_inr * ngn_rate, 2)

    # Log the request into a csv for future retraining pass
    log_dict = input_dict.copy()
   
    log_dict["predicted_price_inr"] = [predicted_inr]
    log_dict["exchange_rate"] = [ngn_rate]
    log_dict["predicted_price_ngn"] = [predicted_ngn]
    log_dict["actual_price"] = [data.actual_price]

    pd.DataFrame(log_dict).to_csv(
        FEEDBACK_FILE, mode="a",
        header=not os.path.exists(FEEDBACK_FILE), index=False,
    )

    return {
        "predicted_price_inr": f"{predicted_inr:,.2f} Indian Rupies",
        "exchange_rate_used": f"1 INR = {ngn_rate:.2f} NGN",
        "predicted_price_ngn": f"{predicted_ngn:,.2f} Naira",
        "status": "Logged successfully",
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT",8000))
    print(f"Loaded model from '{MODEL_PATH}'.")
    print("Starting FAstAPI server on {port} (docs at /docs) ...")
    uvicorn.run(app, host="0.0.0.0", port=port)




