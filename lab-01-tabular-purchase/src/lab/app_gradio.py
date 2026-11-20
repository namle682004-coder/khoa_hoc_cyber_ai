import gradio as gr
import pandas as pd
import joblib
import json
import shap

# ================= LOAD MODEL =================
pipeline = joblib.load("outputs/models/best_model.pkl")

with open("outputs/schema/feature_schema.json") as f:
    schema = json.load(f)

# ================= SHAP =================
model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocessor"]

explainer = shap.TreeExplainer(model)

# ================= PREDICT FUNCTION =================
def predict(
    Administrative,
    Administrative_Duration,
    Informational,
    Informational_Duration,
    ProductRelated,
    ProductRelated_Duration,
    BounceRates,
    ExitRates,
    PageValues,
    SpecialDay,
    Month,
    OperatingSystems,
    Browser,
    Region,
    TrafficType,
    VisitorType,
    Weekend
):

    data = {
        "Administrative": int(Administrative),
        "Administrative_Duration": float(Administrative_Duration),
        "Informational": int(Informational),
        "Informational_Duration": float(Informational_Duration),
        "ProductRelated": int(ProductRelated),
        "ProductRelated_Duration": float(ProductRelated_Duration),
        "BounceRates": float(BounceRates),
        "ExitRates": float(ExitRates),
        "PageValues": float(PageValues),
        "SpecialDay": float(SpecialDay),
        "Month": str(Month),
        "OperatingSystems": str(OperatingSystems),
        "Browser": str(Browser),
        "Region": str(Region),
        "TrafficType": str(TrafficType),
        "VisitorType": str(VisitorType),
        "Weekend": str(Weekend)
    }

    df = pd.DataFrame([data])

    # ================= MODEL PREDICT =================
    prob = pipeline.predict_proba(df)[0][1]
    pred = pipeline.predict(df)[0]

    result = "Customer WILL BUY 🛒" if pred == 1 else "Customer will NOT buy ❌"

    # ================= SHAP =================
    X_trans = preprocessor.transform(df)
    shap_values = explainer.shap_values(X_trans)

    shap_df = pd.DataFrame({
        "feature": preprocessor.get_feature_names_out(),
        "impact": shap_values[0]
    })

    shap_df = shap_df.sort_values("impact", key=abs, ascending=False).head(5)

    explanation = ""
    for _, row in shap_df.iterrows():
        explanation += f"{row['feature']} → impact {row['impact']:.3f}\n"

    return result, round(prob, 4), explanation


# ================= UI =================
with gr.Blocks() as demo:

    gr.Markdown("# 🛒 Online Shopper Purchase Predictor")

    with gr.Row():
        Administrative = gr.Number(value=0)
        Administrative_Duration = gr.Number(value=0)

    with gr.Row():
        Informational = gr.Number(value=0)
        Informational_Duration = gr.Number(value=0)

    with gr.Row():
        ProductRelated = gr.Number(value=0)
        ProductRelated_Duration = gr.Number(value=0)

    with gr.Row():
        BounceRates = gr.Number(value=0)
        ExitRates = gr.Number(value=0)

    with gr.Row():
        PageValues = gr.Number(value=0)
        SpecialDay = gr.Number(value=0)

    Month = gr.Dropdown(
        ["Feb","Mar","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        value="May"
    )

    OperatingSystems = gr.Number(value=1)
    Browser = gr.Number(value=1)
    Region = gr.Number(value=1)
    TrafficType = gr.Number(value=1)

    VisitorType = gr.Dropdown(
        ["Returning_Visitor","New_Visitor","Other"],
        value="Returning_Visitor"
    )

    Weekend = gr.Dropdown(["True","False"], value="False")

    btn = gr.Button("Predict")

    output1 = gr.Textbox(label="Prediction")
    output2 = gr.Textbox(label="Purchase Probability")
    output3 = gr.Textbox(label="Top SHAP Factors")

    btn.click(
        predict,
        inputs=[
            Administrative,
            Administrative_Duration,
            Informational,
            Informational_Duration,
            ProductRelated,
            ProductRelated_Duration,
            BounceRates,
            ExitRates,
            PageValues,
            SpecialDay,
            Month,
            OperatingSystems,
            Browser,
            Region,
            TrafficType,
            VisitorType,
            Weekend
        ],
        outputs=[output1, output2, output3]
    )

demo.launch()