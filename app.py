from flask import Flask, render_template, request
import pandas as pd
from flask import jsonify
import joblib

app = Flask(__name__)

# ===========================
# Load Model
# ===========================

model = joblib.load("model.pkl")
median_size = joblib.load("median_size.pkl")


# ==========================
# Load Dataset
# ==========================

housing_data = pd.read_csv("india_housing_prices.csv")

states = sorted(housing_data["State"].dropna().unique())
cities = sorted(housing_data["City"].dropna().unique())
property_types = sorted(housing_data["Property_Type"].dropna().unique())


# ===========================
# Home Page
# ===========================

@app.route("/get_cities/<state>")
def get_cities(state):

    print("Selected:", state)

    filtered = housing_data[housing_data["State"].str.strip() == state.strip()]

    cities = sorted(filtered["City"].dropna().unique())

    print(cities)

    return jsonify(cities)

@app.route("/")
def home():
    return render_template(
        "index.html",
        states=states,
        cities=cities,
        property_types=property_types
    )


# ===========================
# Prediction
# ===========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        state = request.form["State"]
        city = request.form["City"]
        property_type = request.form["Property_Type"]

        bhk = int(request.form["BHK"])

        size = request.form["Size_in_SqFt"]

        # Optional Size
        if size == "":
            size = median_size
        else:
            size = float(size)

        data = pd.DataFrame({
            "State": [state],
            "City": [city],
            "Property_Type": [property_type],
            "BHK": [bhk],
            "Size_in_SqFt": [size]
        })

        prediction = model.predict(data)[0]
        prediction = round(prediction, 2)

        if prediction >= 100:
            display_price = f"₹ {prediction/100:.2f} Crore"
        else:
            display_price = f"₹ {prediction:.2f} Lakhs"

        return render_template(
           "index.html",
            prediction=display_price,
            states=states,
            cities=cities,
            property_types=property_types
       )

    except Exception as e:

        return render_template(
           "index.html",
            prediction=f"Error: {e}",
            states=states,
            cities=cities,
            property_types=property_types
        )


if __name__ == "__main__":
    app.run(debug=True)