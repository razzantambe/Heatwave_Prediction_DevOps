from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model/heatwave_model.pkl")
FEATURES = ["max_temp", "min_temp", "humidity", "wind_speed", "previous_temp"]

@app.get("/")
def home():
    return "Heatwave Prediction API is running"

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": "Missing fields", "fields": missing}), 400
    try:
        row = pd.DataFrame([[float(data[f]) for f in FEATURES]], columns=FEATURES)
        pred = int(model.predict(row)[0])
    except (TypeError, ValueError):
        return jsonify({"error": "Feature values must be numeric"}), 400
    return jsonify({"prediction": "Heatwave" if pred == 1 else "No Heatwave"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
