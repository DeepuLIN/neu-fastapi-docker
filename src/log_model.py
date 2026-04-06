import mlflow
import mlflow.pytorch

from src.model import LitModel

CHECKPOINT_PATH = "models/mobilenet-best.ckpt"

model = LitModel.load_from_checkpoint(CHECKPOINT_PATH)
model.eval()

mlflow.set_experiment("neu-mlops")

with mlflow.start_run() as run:
    mlflow.pytorch.log_model(model, "model")
    print("Logged model!")
    print("Run ID:", run.info.run_id)