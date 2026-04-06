import mlflow

run_id = "e19c3f2fe1c84ce3a94216e02ebae340"

# Step 1: Register model
result = mlflow.register_model(
    f"runs:/{run_id}/model",
    "neu-defect-model"
)

# Step 2: Set alias = production
client = mlflow.tracking.MlflowClient()

client.set_registered_model_alias(
    name="neu-defect-model",
    alias="production",
    version=result.version
)

print(f"Model registered as version {result.version} and set to 'production'")