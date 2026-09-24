from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.model_loader import ModelLoader



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