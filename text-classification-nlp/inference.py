import argparse
import pickle
import time

from config import TFIDF_PATH


class TfidfModel:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.vectorizer, self.model = pickle.load(f)

    def predict(self, text):
        X = self.vectorizer.transform([text])
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0]
        return pred, prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="tfidf")
    args = parser.parse_args()

    model = TfidfModel(TFIDF_PATH)

    start = time.time()
    label, prob = model.predict(args.text)
    inference_time = time.time() - start

    sentiment = "Positive" if label == 1 else "Negative"

    print("\nPrediction:")
    print("Label:", sentiment)
    print("Confidence:", max(prob))
    print("Inference time:", f"{inference_time*1000:.2f} ms")


if __name__ == "__main__":
    main()