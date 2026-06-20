import os

from fastapi import HTTPException
from google.adk.runners import InMemoryRunner

import app.state as state
from agents import build_fairness_workflow


async def run_bias_workflow(
    dataset_path: str,
    label_col: str,
    pred_col: str = None,
    output_filename: str = "fair_adult_model.pkl"
):
    """
    Orchestrates the bias mitigation workflow using the ADK 2.2.0 graph workflow.
    Automatically reloads the newly generated fair model into the active state on completion.
    """
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail=f"Dataset file '{dataset_path}' not found.")

    try:
        # Build the fairness workflow
        workflow = build_fairness_workflow(
            dataset_path=dataset_path,
            label_col=label_col,
            pred_col=pred_col,
            output_filename=output_filename
        )

        # Initialize runner
        runner = InMemoryRunner(agent=workflow)

        # Execute workflow
        events = await runner.run_debug("Start Bias Detection and Mitigation Pipeline.")

        # Extract mitigation report from logs
        final_report = None
        for event in events:
            if event.author == "MitigationAgent" and event.content and event.content.parts:
                final_report = event.content.parts[0].text

        # Reload model into serving memory
        saved_path = state.MODEL_DIR / output_filename
        state.load_model(saved_path)

        return {
            "status": "success",
            "message": f"Pipeline completed. Fair model saved to: {saved_path}",
            "report": final_report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}") from e
