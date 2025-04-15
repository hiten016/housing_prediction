import os
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('housing_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        area = float(data['area'])
        bedrooms = float(data['bedrooms'])
        bathrooms = float(data['bathrooms'])
        stories = float(data['stories'])
        parking = float(data['parking'])
        furnishingstatus = data['furnishingstatus']

        furnishingstatus_mapping = {'furnished': 1, 'semi-furnished': 2, 'unfurnished': 0}
        furnishingstatus = furnishingstatus_mapping.get(furnishingstatus, 0)  

        features = np.array([[area, bedrooms, bathrooms, stories, parking, furnishingstatus]])

        prediction = model.predict(features)

        predicted_price = np.expm1(prediction[0])

        return jsonify({'predicted_price': predicted_price})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
