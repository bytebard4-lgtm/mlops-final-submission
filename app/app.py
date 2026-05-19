"""
Flask API that loads the trained best model from local storage.
This is the production-safe deployment version.
"""

from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "models/best_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise Exception("Model not found. Run training first.")

print("Loading model from local file...")
model = joblib.load(MODEL_PATH)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = data["features"]

    prediction = model.predict([features])

    return jsonify({
        "prediction": prediction.tolist()
    })


@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": True}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
