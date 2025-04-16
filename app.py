import os
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('housing_price_model.pkl', 'rb'))

furnishingstatus_mapping = {'furnished': 1, 'semi-furnished': 2, 'unfurnished': 0}

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

        furnishingstatus = furnishingstatus_mapping.get(furnishingstatus, 0)  

        features = np.array([[area, bedrooms, bathrooms, stories, parking, furnishingstatus]])

        numeric_features = [area, bedrooms, bathrooms, stories, parking]
        scaler = model.named_steps['preprocessor'].transformers_[0][1]
        scaled_numeric_features = scaler.transform([numeric_features])

        encoder = model.named_steps['preprocessor'].transformers_[1][1]
        encoded_furnishingstatus = encoder.transform([[furnishingstatus]]).toarray()

        preprocessed_features = np.concatenate([scaled_numeric_features, encoded_furnishingstatus], axis=1)

        prediction = model.predict(preprocessed_features)

        predicted_price = np.expm1(prediction[0])

        return jsonify({'predicted_price': predicted_price})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
