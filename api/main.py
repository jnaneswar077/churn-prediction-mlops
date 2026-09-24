from contextlib import asynccontextmanager
from fastapi import FastAPI
import pandas as pd

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data
from api.schemas import CustomerInput

model_loader = ModelLoader()


# @app.on_event("startup")
# def startup_event():
#     model_loader.load()

@asynccontextmanager
async def lifespan(app: FastAPI):
    #startup
    model_loader.load()

    yield

    #shutdowon
    print("[INFO] Churn Prediction API shutting down...")


app = FastAPI(
    title="Churn Prediction API",
    description="Lab 10 - ML Model Serving and API Engineering",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is running",
        "lab": "Lab 10",
        "status": "success",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.get("/metadata")
def metadata():

    return {
        "service": "Churn Prediction API",
        "version": "1.0.0",
        "model_name": model_loader.model_name,
        "model_uri": model_loader.model_uri,
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None,
    }

@app.post("/predict")
def predict(data: CustomerInput):


    # 1. Convert JSON data into a DataFrame
    # df = pd.DataFrame([data])

    # Convert Pydantic object to dictionary
    data_dict = data.model_dump()

    # Convert dictionary to DataFrame
    df = pd.DataFrame([data_dict])
    
    # 2. Clean the raw input
    df = clean_prediction_data(df)

    # 3. Transform using the saved preprocessing pipeline
    X = model_loader.preprocessor.transform(df)

    # 4. Generate prediction
    prediction = model_loader.model.predict(X)[0]

    # 5. Generate churn probability
    probability = model_loader.model.predict_proba(X)[0][1]

    # 6. Return the prediction
    return {
        "prediction": int(prediction),
        "churn_probability": float(probability),
    }