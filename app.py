from flask import Flask, render_template, request
import joblib
import numpy as np

# For real-world fare estimator
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

app = Flask(__name__)

# Load NYC ML model
model = joblib.load("decision_tree_model.pkl")
feature_names = joblib.load("model_features.pkl")

# -------------------------
# Route 1: NYC ML prediction
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Get form data
            data = [float(request.form[feature]) for feature in feature_names]
            array = np.array(data).reshape(1, -1)
            prediction = round(model.predict(array)[0], 2)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("index.html", prediction=prediction, features=feature_names)

# ---------------------------------------
# Route 2: Real-World Fare Estimator
# ---------------------------------------
@app.route("/realworld", methods=["GET", "POST"])
def realworld():
    fare_result = None
    if request.method == "POST":
        pickup = request.form["pickup"]
        dropoff = request.form["dropoff"]

        geolocator = Nominatim(user_agent="fare_predictor", timeout=10)
        loc1 = geolocator.geocode(pickup)
        loc2 = geolocator.geocode(dropoff)

        if loc1 and loc2:
            coord1 = (loc1.latitude, loc1.longitude)
            coord2 = (loc2.latitude, loc2.longitude)

            distance_km = geodesic(coord1, coord2).km
            duration_min = distance_km  # assume 60 km/hr

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
            fare_result = "Invalid location(s). Please try again."

    return render_template("realworld.html", result=fare_result)

# -------------------
if __name__ == "__main__":
    app.run(debug=True)
