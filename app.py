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
    data = request.get_json()

    area = data['area']
    bedrooms = data['bedrooms']
    bathrooms = data['bathrooms']
    stories = data['stories']
    parking = data['parking']
    furnishingstatus = data['furnishingstatus']
    mainroad = data['mainroad']
    guestroom = data['guestroom']
    basement = data['basement']
    hotwaterheating = data['hotwaterheating']
    airconditioning = data['airconditioning']

    features = np.array([[area, bedrooms, bathrooms, stories, parking,
                          furnishingstatus, mainroad, guestroom, basement, hotwaterheating, airconditioning]])

    prediction = model.predict(features)

    predicted_price = np.expm1(prediction[0])

    return jsonify({'predicted_price': predicted_price})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port, debug=True)
