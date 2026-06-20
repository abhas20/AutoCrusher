from fastapi import APIRouter
from pydantic import BaseModel

from app.controllers.prediction_controller import predict_fair_outcome

router = APIRouter(prefix="/predict", tags=["Prediction"])

class PredictionRequest(BaseModel):
    features: dict
    model_filename: str = "fair_adult_model.pkl"

@router.post("/", summary="Serve fair predictions from the mitigated model")
def get_prediction(request: PredictionRequest):
    """
    Exposes prediction endpoint. Expects a JSON object containing the features
    for the model (e.g. age, workclass, education, etc. for the Adult dataset).
    Example JSON input:
    {
        "features": {
            "age": 25, "workclass": "Private", "education": "Bachelors"
        },
        "model_filename": "fair_adult_model.pkl"
    }

    """
    return predict_fair_outcome(request.features,request.model_filename)
