import pickle
from flask import Flask
from flask import request
from flask import jsonify
import json
import numpy as np

input_file = '../model/ridge_model.bin'

with open(input_file, 'rb') as f_in: 
    model, test_data = pickle.load(f_in)

app = Flask('Food Delivery Time Prediction')

@app.route('/predict', methods=['POST'])
def predict_price():
    deliveryInfo = request.get_json()
    time_pred = float(model.predict(deliveryInfo)[0])
    
    result = {
        'time_prediction': time_pred
    }

    # return 'The predicted delivery time is: ' + str(time_pred)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=9696, debug=True, host='0.0.0.0')


def convert_numpy_bools(data):
    """Recursively converts numpy.bool_ to bool in dictionaries and lists."""
    if isinstance(data, dict):
        return {key: convert_numpy_bools(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_bools(item) for item in data]
    elif isinstance(data, np.bool_):  # Check for NumPy boolean type
        return bool(data)  # Convert to standard Python boolean
    return data  # Return other data types as is
