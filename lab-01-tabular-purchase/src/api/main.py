from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json
import os

from sqlalchemy import true

app = FastAPI(title="Purchase Prediction API")

# ---------- Load Model ----------
MODEL_PATH = "outputs/models/best_model.pkl"
SCHEMA_PATH = "outputs/schema/feature_schema.json"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model not found. Run train.py first.")

model = joblib.load(MODEL_PATH)

# ---------- Load Feature Schema ----------
with open(SCHEMA_PATH) as f:
    schema = json.load(f)

columns = list(schema.keys())


# ---------- Request Schema ----------
class PurchaseRequest(BaseModel):

    Administrative: int
    Administrative_Duration: float
    Informational: int
    Informational_Duration: float
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float

    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool


# ---------- Health Check ----------
@app.get("/")
def root():
    return {"status": "ok"}


# ---------- Example Data ----------
@app.get("/unbuy")
def example():

    return {
        "Administrative": 3,
        "Administrative_Duration": 45.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 20,
        "ProductRelated_Duration": 300.5,
        "BounceRates": 0.02,
        "ExitRates": 0.05,
        "PageValues": 10.0,
        "SpecialDay": 0.0,
        "Month": "Nov",
        "OperatingSystems": 2,
        "Browser": 1,
        "Region": 3,
        "TrafficType": 2,
        "VisitorType": "Returning_Visitor",
        "Weekend": False
    }

@app.get("/buy")
def example():

    return {
          "Administrative": 5,
          "Administrative_Duration": 300,
          "Informational": 4,
          "Informational_Duration": 200,
          "ProductRelated": 120,
          "ProductRelated_Duration": 5000,
          "BounceRates": 0.001,
          "ExitRates": 0.005,
          "PageValues": 80,
          "SpecialDay": 0,
          "Month": "Nov",
          "OperatingSystems": 2,
          "Browser": 2,
          "Region": 1,
          "TrafficType": 2,
          "VisitorType": "Returning_Visitor",
          "Weekend": true
    }



# ---------- Prediction ----------
@app.post("/predict")
def predict(data: PurchaseRequest):

    # convert request -> dataframe
    df = pd.DataFrame([data.model_dump()])

    # ---------- FORCE DTYPE FROM SCHEMA ----------
    for col, dtype in schema.items():

        if dtype == "int64":
            df[col] = df[col].astype(int)

        elif dtype == "float64":
            df[col] = df[col].astype(float)

        elif dtype == "category":
            df[col] = df[col].astype(str)

    # ---------- Predict ----------
    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    return {
        "prediction": int(pred),
        "purchase_probability": float(prob)
    }