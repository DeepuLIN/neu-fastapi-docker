# eport_pdf.py

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import os


mlflow.set_tracking_uri("sqlite:///mlflow.db")

experiment = mlflow.get_experiment_by_name("neu-classification")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

df = runs[[
    "params.model_name",
    "metrics.val_loss",
    "metrics.val_acc"
]]

df = df.sort_values(by="metrics.val_loss")




plt.figure()
plt.bar(df["params.model_name"], df["metrics.val_acc"])
plt.title("Validation Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.savefig("accuracy_plot.png")
plt.close()


plt.figure()
plt.bar(df["params.model_name"], df["metrics.val_loss"])
plt.title("Validation Loss")
plt.xlabel("Model")
plt.ylabel("Loss")
plt.savefig("loss_plot.png")
plt.close()

doc = SimpleDocTemplate("MLflow_Report.pdf", pagesize=A4)
styles = getSampleStyleSheet()

content = []


content.append(Paragraph("NEU Surface Defect Classification Report", styles["Title"]))
content.append(Spacer(1, 20))


best_model = df.iloc[0]

summary_text = f"""
Best Model: <b>{best_model['params.model_name']}</b><br/>
Validation Loss: {best_model['metrics.val_loss']:.4f}<br/>
Validation Accuracy: {best_model['metrics.val_acc']:.4f}
"""

content.append(Paragraph(summary_text, styles["Normal"]))
content.append(Spacer(1, 20))

content.append(Paragraph("Model Accuracy Comparison", styles["Heading2"]))
content.append(Image("accuracy_plot.png", width=400, height=250))
content.append(Spacer(1, 20))

content.append(Paragraph("Model Loss Comparison", styles["Heading2"]))
content.append(Image("loss_plot.png", width=400, height=250))

doc.build(content)

print("✅ PDF report generated: MLflow_Report.pdf")