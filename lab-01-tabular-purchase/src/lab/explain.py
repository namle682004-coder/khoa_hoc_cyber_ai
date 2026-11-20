import joblib
import shap
import matplotlib.pyplot as plt

from src.lab.data import load_data

print("Loading model...")
pipeline = joblib.load("outputs/models/best_model.pkl")

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

print("Loading data...")
X, y = load_data()

print("Transforming data...")
X_transformed = preprocessor.transform(X)

# convert sparse → dense
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

# convert dtype
X_transformed = X_transformed.astype(float)

print("Building SHAP explainer...")
explainer = shap.TreeExplainer(model)

print("Computing SHAP values...")
shap_values = explainer.shap_values(X_transformed)

print("Generating plots...")

shap.summary_plot(shap_values, X_transformed, show=False)
plt.savefig("outputs/reports/shap_summary.png", bbox_inches="tight")

shap.summary_plot(shap_values, X_transformed, plot_type="bar", show=False)
plt.savefig("outputs/reports/shap_bar.png", bbox_inches="tight")

print("SHAP plots saved.")