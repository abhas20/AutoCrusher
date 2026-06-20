from fastapi import APIRouter
from pydantic import BaseModel

from app.controllers.mitigation_controller import run_bias_workflow

router = APIRouter(prefix="/run-mitigation", tags=["Mitigation"])

class AuditRequest(BaseModel):
    dataset_path: str
    label_col: str
    pred_col: str = None
    output_filename: str = "fair_adult_model.pkl"

@router.post("/", summary="Run bias detection and mitigation pipeline")
async def start_mitigation(request: AuditRequest):
    """
    Exposes mitigation pipeline trigger endpoint. Expects a JSON object specifying
    the path to the CSV file, label column, optional prediction column, and output filename.
    """
    return await run_bias_workflow(
        dataset_path=request.dataset_path,
        label_col=request.label_col,
        pred_col=request.pred_col,
        output_filename=request.output_filename
    )
