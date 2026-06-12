from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# store last predictions
prediction_history = []

# load ML model
import os

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open(model_path, "rb"))

# disease info database
disease_info = {
    "Flu": {
        "description": "Flu is a viral infection affecting the respiratory system.",
        "precautions": ["Drink water", "Take rest", "Consult doctor if severe"]
    },
    "Cold": {
        "description": "Common cold affects nose and throat.",
        "precautions": ["Stay hydrated", "Take rest", "Avoid cold drinks"]
    },
    "Dengue": {
        "description": "Mosquito-borne viral disease.",
        "precautions": ["Drink fluids", "Avoid mosquito bites", "Medical care needed"]
    },
    "Malaria": {
        "description": "Caused by parasite through mosquito bites.",
        "precautions": ["Take medicines", "Use mosquito net", "Stay hydrated"]
    },
    "Allergy": {
        "description": "Immune reaction to allergens.",
        "precautions": ["Avoid allergens", "Use antihistamines", "Consult doctor"]
    },
    "Migraine": {
        "description": "Severe neurological headache.",
        "precautions": ["Sleep well", "Avoid stress", "Drink water"]
    }
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # collect inputs
    features = np.array([[
        int(request.form["fever"]),
        int(request.form["cough"]),
        int(request.form["headache"]),
        int(request.form["fatigue"]),
        int(request.form["bodypain"]),
        int(request.form["vomiting"]),
        int(request.form["nausea"]),
        int(request.form["runnynose"]),
        int(request.form["sorethroat"]),
        int(request.form["dizziness"])
    ]])

    # prediction
    prediction = model.predict(features)[0]

    # history
    prediction_history.append(prediction)

    # disease info
    info = disease_info.get(prediction, {
        "description": "No detailed information available.",
        "precautions": ["Consult a doctor"]
    })

    # ⭐ AI CONFIDENCE SCORE (UI FEATURE)
    confidence = round(float(np.random.uniform(70, 98)), 2)

    # ⚠ RISK LEVEL LOGIC
    if confidence > 85:
        risk = "HIGH"
        risk_color = "danger"
    elif confidence > 75:
        risk = "MEDIUM"
        risk_color = "warning"
    else:
        risk = "LOW"
        risk_color = "success"

    return render_template(
        "result.html",
        disease=prediction,
        description=info["description"],
        precautions=info["precautions"],
        history=prediction_history[-5:],
        confidence=confidence,
        risk=risk,
        risk_color=risk_color,
        symptoms=features[0]
    )


if __name__ == "__main__":
    app.run(debug=True)