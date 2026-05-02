from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

import os

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None


@app.route("/")
def home():
    return {"message": "Flask ML App is working!"}


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)

    return {"prediction": int(prediction[0])}


if __name__ == "__main__":
    app.run(debug=True)