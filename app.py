from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

from recommendation import recommend
from analytics import analyze_data
from insights import get_insight

app = Flask(__name__)

# Load Data and Model
with open("df.pkl", "rb") as file:
    df = pickle.load(file)

with open("pipeline.pkl", "rb") as file:
    pipeline = pickle.load(file)
    
# print("hello2")
@app.route("/")
def home():
    """Render the home page with dropdown options."""
    # print("hello3")
    return render_template(
        "index.html",
        sectors=sorted(df["sector"].unique().tolist()),
        bedrooms=sorted(df["bedRoom"].unique().tolist()),
        bathrooms=sorted(df["bathroom"].unique().tolist()),
        balconies=sorted(df["balcony"].unique().tolist()),
        property_ages=sorted(df["agePossession"].unique().tolist()),
        furnishing_types=sorted(df["furnishing_type"].unique().tolist()),
        luxury_categories=sorted(df["luxury_category"].unique().tolist()),
        floor_categories=sorted(df["floor_category"].unique().tolist()),
    
    )

@app.route("/predict", methods=["POST" , "GET"])
def predict():
    """Predicts the price of the property based on user input."""
    try:
        data = request.form
        # print("hello4")

        # Ensure all required fields are present
        required_fields = [
            "property_type", "sector", "bedroom", "bathroom", "balcony",
            "property_age", "built_up_area", "servant_room", "store_room",
            "furnishing_type", "luxury_category", "floor_category"
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        # Convert inputs to the correct type
        try:
            input_data = [[
                data["property_type"],
                data["sector"],
                int(data["bedroom"]),   # Ensure integer
                int(data["bathroom"]),  # Ensure integer
                data["balcony"],
                data["property_age"],
                float(data["built_up_area"]),  # Ensure float
                int(data["servant_room"]),  # Ensure integer
                int(data["store_room"]),  # Ensure integer
                data["furnishing_type"],
                data["luxury_category"],
                data["floor_category"]
            ]]
        except ValueError as ve:
            return jsonify({"error": f"Invalid input type: {ve}"}), 400

        # Convert input to DataFrame
        columns = [
            "property_type", "sector", "bedRoom", "bathroom", "balcony",
            "agePossession", "built_up_area", "servant room", "store room",
            "furnishing_type", "luxury_category", "floor_category"
        ]
        one_df = pd.DataFrame(input_data, columns=columns)
        # print("Received Data:", data)
        print(input_data)

        # Predict
        predicted_price = pipeline.predict(one_df)
        # print("hello6")
        # print("Received Data:", data)
        # print("Predicted Price:", predicted_price)
        # print("hello6")
        base_price = np.expm1(predicted_price)[0]  # Apply inverse log transformation

        # Compute the range
        low = round(base_price - 0.22, 2)
        high = round(base_price + 0.22, 2)
        # print("hello7")

        return jsonify({"prediction": f"The price of the flat is between {low} Cr and {high} Cr"})

    except Exception as e:
        app.logger.error(f"Error in prediction: {e}")
        return jsonify({"error": "An internal error occurred. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True)
