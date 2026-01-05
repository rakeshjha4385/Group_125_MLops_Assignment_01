from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
from fastapi import Request
from prometheus_fastapi_instrumentator import Instrumentator


class PatientFeatures(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float


app = FastAPI()

model = joblib.load("models/random_forest.joblib")
pipeline = joblib.load("models/preprocessing_pipeline.joblib")

Instrumentator().instrument(app).expose(app)


@app.post("/predict")
def predict(features: PatientFeatures):
    X = pd.DataFrame(
        [
            {
                "age": features.age,
                "sex": features.sex,
                "cp": features.cp,
                "trestbps": features.trestbps,
                "chol": features.chol,
                "fbs": features.fbs,
                "restecg": features.restecg,
                "thalach": features.thalach,
                "exang": features.exang,
                "oldpeak": features.oldpeak,
                "slope": features.slope,
                "ca": features.ca,
                "thal": features.thal,
            }
        ]
    )

    X_prep = pipeline.transform(X)
    prediction = model.predict(X_prep)[0]
    probability = model.predict_proba(X_prep)[0][1]

    return {"prediction": int(prediction), "confidence": float(probability)}


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart_api")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response
