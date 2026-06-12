from google.adk import Agent, Workflow
from google.adk.models import Gemini
from tools import analyze_dataset_fields, calculate_fairlearn_bias, apply_bias_mitigation

def build_fairness_workflow(
    dataset_path: str,
    label_col: str,
    pred_col: str = None,
    output_filename: str = "fair_model.pkl",
) -> Workflow:
    """    
    1. AuditorAgent: Audits columns to find useless fields and sensitive proxy candidates.
    2. BiasConfirmerAgent: Confirms bias on candidates using Fairlearn metrics.
    3. MitigationAgent: Retrains the model under fairness constraints and exports the PKL.
    """
    if not pred_col:
        from tools import generate_baseline_predictions
        dataset_path, pred_col = generate_baseline_predictions(dataset_path, label_col)
    
    # ── Agent 1: AuditorAgent ─────────────────────────────────────────────────
    auditor_agent = Agent(
        name="AuditorAgent",
        model=Gemini(model="gemma-4-31b-it"),
        description="Audits dataset fields to find useless/unnecessary fields and sensitive proxy candidates.",
        instruction=f"""
      You are an AI Data Auditor. Your task is to identify:
        1. Useless or unnecessary fields (constant columns, unique IDs, or high missingness).
        2. Potential sensitive proxy candidates.

      Call the tool `analyze_dataset_fields` with arguments:
        - dataset_path = "{dataset_path}"
        - label_col    = "{label_col}"
        - pred_col     = "{pred_col}"

      Analyze the tool's output. Write a clean summary of what fields are useless and which are sensitive candidates.
      Output ONLY a JSON block summarizing the findings under these keys:
        - useless_fields: list of columns identified as useless
        - sensitive_candidates: list of columns identified as sensitive proxy candidates

      Format:
      {{
        "useless_fields": ["col_1", "col_2"],
        "sensitive_candidates": ["col_3", "col_4"]
      }}
      """,
        tools=[analyze_dataset_fields],
        output_key="audit_report",
    )

    # ── Agent 2: BiasConfirmerAgent ───────────────────────────────────────────
    bias_confirmer_agent = Agent(
        name="BiasConfirmerAgent",
        model=Gemini(model="gemma-4-31b-it"),
        description="Confirms bias mathematically using Fairlearn metrics.",
        instruction=f"""
You are an AI Bias Confirmer. Read the audit report from the previous step:
{{audit_report}}

From the audit report, extract the list of "sensitive_candidates".
If the list is empty, write a response indicating no sensitive candidates were found.

For EACH sensitive candidate in that list:
  Call `calculate_fairlearn_bias` with:
    - dataset_path = "{dataset_path}"
    - label_col    = "{label_col}"
    - pred_col     = "{pred_col}"
    - sensitive_feature_name = <column name>

Collect and analyze the results. Output ONLY a JSON block with:
  - biased_columns: list of columns where overall_verdict is BIASED
  - cleared_columns: list of columns where overall_verdict is FAIR
  - scores: dict of scores for each tested column

Format:
{{
  "biased_columns": ["col_3"],
  "cleared_columns": ["col_4"],
  "scores": {{
    "col_3": {{
      "demographic_parity_difference": 0.21,
      "disparate_impact_ratio": 0.55,
      "equalized_odds_difference": 0.12,
      "verdict": "BIASED"
    }}
  }}
}}
""",
        tools=[calculate_fairlearn_bias],
        output_key="bias_report",
    )

    # ── Agent 3: MitigationAgent ───────────────────────────────────────────────
    mitigation_agent = Agent(
        name="MitigationAgent",
        model=Gemini(model="gemma-4-31b-it"),
        description="Mitigates bias and retrains fair models.",
        instruction=f"""
You are an AI Model Mitigator. Read the following from session state:
  - audit_report: {{audit_report}}
  - bias_report: {{bias_report}}

Steps:
  1. Extract "useless_fields" from audit_report.
  2. Extract "biased_columns" and "scores" from bias_report.
  3. If "biased_columns" is empty, write a report showing that no bias mitigation is necessary since no columns are biased.
  4. If there are biased columns, identify the primary sensitive column: this is the biased column with the highest demographic_parity_difference in the scores.
  5. Call the tool `apply_bias_mitigation` with:
       - dataset_path = "{dataset_path}"
       - label_col    = "{label_col}"
       - biased_columns = <JSON string list of all biased columns, e.g. ["col_3"]>
       - primary_sensitive_col = <primary sensitive column name>
       - pred_col     = "{pred_col}"
       - output_filename = "{output_filename}"

Print the final mitigation summary:
  - List the useless fields that were dropped.
  - List the biased fields that were mitigated.
  - The best model trained (e.g. RandomForest) and its performance metrics (accuracy, fairness, composite score).
  - The path of the exported .pkl model.
""",
        tools=[apply_bias_mitigation],
        output_key="mitigation_report",
    )

    # ── Define the workflow ───────────────────────────────────────────────────
    workflow = Workflow(
        name="FairnessWorkflow",
        edges=[
            ("START", auditor_agent, bias_confirmer_agent, mitigation_agent)
        ]
    )
    
    return workflow
