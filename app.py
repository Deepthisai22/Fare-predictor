from flask import Flask, render_template, request
import joblib
import numpy as np
import googlemaps
import os
from dotenv import load_dotenv
load_dotenv()

gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))

# Load your trained ML model
model = joblib.load("decision_tree_model.pkl")
feature_names = joblib.load("model_features.pkl")

# Initialize Flask app
app = Flask(__name__)

# Initialize Google Maps client with your API key
gmaps = googlemaps.Client(key="AIzaSyC2nPpAXrrxj7Uyqec9mN1yHn7NHyh5Ul0")  # GCS API KEY

# -------------------------
# Route 1: ML Prediction (NYC)
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            data = [float(request.form[feature]) for feature in feature_names]
            array = np.array(data).reshape(1, -1)
            prediction = round(model.predict(array)[0], 2)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("index.html", prediction=prediction, features=feature_names)


# -------------------------
# Route 2: Real-World Fare Estimator
# -------------------------
@app.route("/realworld", methods=["GET", "POST"])
def realworld():
    fare_result = None
    if request.method == "POST":
        pickup = request.form["pickup"]
        dropoff = request.form["dropoff"]

        try:
            result = gmaps.distance_matrix(pickup, dropoff, mode="driving")

            # Check for valid result
            if result["rows"][0]["elements"][0]["status"] == "OK":
                distance_m = result["rows"][0]["elements"][0]["distance"]["value"]
                duration_s = result["rows"][0]["elements"][0]["duration"]["value"]

                distance_km = distance_m / 1000
                duration_min = duration_s / 60

                # Example fare logic
                base_fare = 50
                fare_per_km = 12
                fare_per_min = 1.5
                total_fare = base_fare + (distance_km * fare_per_km) + (duration_min * fare_per_min)

                fare_result = {
                    "pickup": pickup.title(),
                    "dropoff": dropoff.title(),
                    "distance": round(distance_km, 2),
                    "duration": round(duration_min, 2),
                    "fare": round(total_fare, 2)
                }
            else:
                fare_result = "Location not found. Please check the address and try again."

        except Exception as e:
            fare_result = f"Error: {e}"

    return render_template("realworld.html", result=fare_result)


# -------------------------
# Run the Flask App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
