import asyncio
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

# ── Mock Biased Dataset ──────────────────────────────────────
def generate_dataset():
    print("⚙️ Generating historical biased dataset...")
    np.random.seed(42)

    # Features that should matter
    income = np.random.randint(30000, 120000, 1000)
    credit_score = np.random.randint(500, 800, 1000)

    # Hidden proxy feature (Zip Code)
    zip_codes = np.random.choice(['10001', '10002'], 1000)

    # Inject bias: Zip code '10001' has higher approval chances
    approval_chance = (income / 1000) + (credit_score / 10)
    bias_modifier = np.where(zip_codes == '10001', 50, -50)
    final_score = approval_chance + bias_modifier

    threshold = np.median(final_score)
    approved = (final_score >= threshold).astype(int)

    df = pd.DataFrame({
        'Income': income,
        'Credit_Score': credit_score,
        'Zip_Code': zip_codes,
        'Approved': approved
    })

    # Prepare features and target
    X = df[['Income', 'Credit_Score', 'Zip_Code']]
    y = df['Approved']

    # Zip_Code as category
    X_train = X.copy()
    X_train['Zip_Code'] = X_train['Zip_Code'].astype('category').cat.codes

    # Train a biased model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y)

    df['Approved_Pred'] = model.predict(X_train)

    csv_path = 'dataset/historical_biased_dataset.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved mock dataset to: {csv_path}")
    return csv_path


# def test_real_dataset():
#     df= pd.read_csv('dataset/adult.csv')
#     for col in df.columns:
#         print(f"{col}: {df[col].unique()[:5]}")




# ── Step 2: Run Fairness Workflow ─────────────────────────────────────────────
async def run_workflow(csv_path: str, label_col: str, pred_col: str = None, output_filename: str = "fair_model.pkl"):
    from google.adk.runners import InMemoryRunner

    from agents import build_fairness_workflow

    print(f"\n🚀 Initializing ADK Fairness Workflow for: {csv_path} (label: {label_col})...")
    workflow = build_fairness_workflow(
        dataset_path=csv_path,
        label_col=label_col,
        pred_col=pred_col,
        output_filename=output_filename
    )

    runner = InMemoryRunner(agent=workflow)
    print("🏃 Running workflow nodes...")

    events = await runner.run_debug("Start Bias Detection and Mitigation Pipeline.")

    print("\n==========================================")
    print("          WORKFLOW EXECUTION LOGS         ")
    print("==========================================\n")

    for event in events:
        if event.content and event.content.parts:
            text = event.content.parts[0].text
            print(f"[{event.author}]:")
            print(text)
            print("-" * 50)


if __name__ == "__main__":
    adult_csv_path = "dataset/adult.csv"
    asyncio.run(run_workflow(
        csv_path=adult_csv_path,
        label_col="income",
        pred_col=None,
        output_filename="fair_adult_model.pkl"
    ))

