
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('housing_price_model', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    area = float(data['area'])
    bedrooms = int(data['bedrooms'])
    bathrooms = int(data['bathrooms'])
    stories = int(data['stories'])
    mainroad = 1 if data.get('mainroad') == 'yes' else 0
    guestroom = 1 if data.get('guestroom') == 'yes' else 0
    basement = 1 if data.get('basement') == 'yes' else 0
    hotwaterheating = 1 if data.get('hotwaterheating') == 'yes' else 0
    airconditioning = 1 if data.get('airconditioning') == 'yes' else 0
    parking = int(data['parking'])

    furnishing_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
    furnishingstatus = furnishing_map[data['furnishingstatus'].lower()]

    preferred_map = {'no': 0, 'yes': 1}
    prefarea = preferred_map[data['prefarea'].lower()]

    input_data = np.array([[area, bedrooms, bathrooms, stories, mainroad,
                            guestroom, basement, hotwaterheating,
                            airconditioning, parking, prefarea, furnishingstatus]])
    prediction = model.predict(input_data)[0]
    return render_template('index.html', prediction_text=f'Predicted Price: ₹{int(prediction):,}')

if __name__ == '__main__':
    app.run(debug=True)
