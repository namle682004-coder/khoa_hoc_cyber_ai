import argparse
import joblib
import yaml
from pathlib import Path
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType


def load_config(path: str):
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    model_path = Path("outputs/models/best_model.pkl")

    if not model_path.exists():
        raise FileNotFoundError("Run training first.")

    print("Loading model...")
    pipeline = joblib.load(model_path)

    if "preprocessor" not in pipeline.named_steps:
        raise ValueError("Pipeline missing 'preprocessor' step.")

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = list(pipeline.feature_names_in_)

    numeric_cols = None
    categorical_cols = None

    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            numeric_cols = cols
        elif name == "cat":
            categorical_cols = cols

    initial_type = []

    for col in feature_names:
        if col in numeric_cols:
            initial_type.append((col, FloatTensorType([None, 1])))
        elif col in categorical_cols:
            initial_type.append((col, StringTensorType([None, 1])))

    print("Converting to ONNX...")
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type
    )

    export_dir = Path("outputs/export")
    export_dir.mkdir(parents=True, exist_ok=True)

    export_path = export_dir / "model.onnx"

    with open(export_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"ONNX model saved to {export_path}")


if __name__ == "__main__":
    main()