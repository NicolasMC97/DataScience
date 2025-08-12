import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictRequest, PredictResponse, HealthResponse
from .model import load_model, predict_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("knn-api")

app = FastAPI(title="KNN Location API", version="1.0.0")

# Optional: restrict in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.on_event("startup")
def startup_event():
    try:
        load_model()
        logger.info("Model loaded successfully at startup.")
    except Exception as e:
        logger.exception("Failed to load model at startup.")
        # Let the app start; /health will show failure if needed

@app.get("/health", response_model=HealthResponse)
def health():
    try:
        load_model()
        return HealthResponse(status="ok")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"model_load_error: {e}")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        label, proba = predict_label(req.latitude, req.longitude)
        return PredictResponse(label=label, probability=proba)
    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=400, detail=f"prediction_error: {e}")
